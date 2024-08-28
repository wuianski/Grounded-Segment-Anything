import argparse
import os
from pathlib import Path
import copy

import numpy as np
import json
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tag2Text
from ram.models import tag2text
from ram import inference_tag2text
import torchvision.transforms as TS

from datetime import datetime


# ChatGPT or nltk is required when using captions
# import openai
# import nltk

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def generate_caption(raw_image, device):
    # unconditional image captioning
    if device == "cuda":
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    else:
        inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_tags(caption, split=',', max_tokens=100, model="gpt-3.5-turbo"):
    lemma = nltk.wordnet.WordNetLemmatizer()
    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Extract the unique nouns in the caption. Remove all the adjectives. ' + \
                           f'List the nouns in singular form. Split them by "{split} ". ' + \
                           f'Caption: {caption}.'
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "noun: xxx, xxx, xxx"
        tags = reply.split(':')[-1].strip()
    else:
        nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])
        tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == 'N']
        tags_lemma = [lemma.lemmatize(w) for w in tags_list]
        tags = ', '.join(map(str, tags_lemma))
    return tags


def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the caption if it is wrong. ' + \
                           f'Caption: {caption}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised caption: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "Caption: xxx, xxx, xxx"
        caption = reply.split(':')[-1].strip()
    return caption


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    boxedModel = model.to(device)
    boxedImage = image.to(device)
    with torch.no_grad():
        outputs = boxedModel(boxedImage[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = boxedModel.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(caption, mask_list, boxes_filt, label_list, results_dir, temp_dir):
    # get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # convert datetime obj to string
    str_current_datetime = str(current_datetime)

    value = 0  # 0 for background
    color = np.array([0 / 255, 0 / 255, 0 / 255, 1.0])

    # original = Image.open(os.path.join(temp_dir, f"automatic_nolabel_output.jpg"))
    for (mask, label, box) in zip(mask_list, label_list, boxes_filt):
        value += 1
        name, logit = label.split('(')
        box = box.numpy()
        # make dir
        img_mask_dir = os.path.join(results_dir, f'img_mask/{name[0].lower()}/')
        # mask
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.numpy()

        ### binary mask ###
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_image)
        plt.axis('off')
        file_name = f"mask_temp_{value}.png"
        plt.savefig(
            os.path.join(temp_dir, file_name),
            bbox_inches="tight", pad_inches=0.0
        )

        ### crop mask_image using box ###
        # crop_img = mask_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # plt.figure(figsize=(10, 10))
        # plt.imshow(crop_img)
        # plt.axis('off')
        # file_name = f"crop_temp_{value}.png"
        # plt.savefig(
        #     os.path.join(temp_dir, file_name),
        #     bbox_inches="tight", pad_inches=0.0
        # )

        ### object + color mask without background ###
        mask_temp = cv2.imread(os.path.join(
            temp_dir, f"mask_temp_{value}.png")
        )
        img = cv2.cvtColor(mask_temp, cv2.COLOR_BGR2GRAY)
        mask_invert = cv2.bitwise_not(img)
        original = cv2.imread(os.path.join(temp_dir, f"automatic_nolabel_output.jpg"))
        object_img_trans = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        object_img_trans[:, :, 3] = mask_invert
        os.makedirs(img_mask_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(img_mask_dir, f"{name}_{str_current_datetime}_{value}.png"),
            object_img_trans
        )

    json_data = {
        'caption': caption,
        'mask': [{
            'value': value,
            'label': 'background'
        }]
    }

    for label, box in zip(label_list, boxes_filt):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(temp_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


def save_mask_crop(origin_size, mask_list, boxes_list, label_list, temp_dir, results_dir, crop_dir, caption,
                   boxes_filt_j):
    # get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # convert datetime obj to string
    str_current_datetime = str(current_datetime)

    org_h, org_w = origin_size
    masked_img = cv2.imread(os.path.join(temp_dir, f"automatic_nolabel_output.jpg"))
    no_mask_img = cv2.imread(os.path.join(temp_dir, f"no_mask.jpg"))
    masked_h, masked_w, _ = masked_img.shape
    scale_y, scale_x = (masked_h / org_h, masked_w / org_w)
    crop_idx = 0
    color = np.array([0 / 255, 0 / 255, 0 / 255, 1.0])

    for mask, box, label in zip(mask_list, boxes_list, label_list):
        crop_idx += 1
        name, logit = label.split('(')
        # make dir
        img_mask_dir = os.path.join(results_dir, f'crop/masked/{name[0].lower()}/')
        img_no_mask_dir = os.path.join(results_dir, f'crop/no_mask/{name[0].lower()}/')
        # round box to int
        int_box = box.astype(int)
        x0, y0, x1, y1 = int_box
        x0, y0, x1, y1 = int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y)
        # mask
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.numpy()

        ### binary mask ###
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_image)
        plt.axis('off')
        file_name = f"mask_temp_{crop_idx}.png"
        plt.savefig(
            os.path.join(temp_dir, file_name),
            bbox_inches="tight", pad_inches=0.0
        )

        mask_temp = cv2.imread(os.path.join(temp_dir, f"mask_temp_{crop_idx}.png"))
        img = cv2.cvtColor(mask_temp, cv2.COLOR_BGR2GRAY)
        mask_invert = cv2.bitwise_not(img)
        # crop masked image
        object_img_trans = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
        object_img_trans[:, :, 3] = mask_invert
        crop_img = object_img_trans[y0:y1, x0:x1, :]
        os.makedirs(img_mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_mask_dir, f"{name}_{str_current_datetime}.png"), crop_img)
        # crop no mask image
        object_img2_trans = cv2.cvtColor(no_mask_img, cv2.COLOR_BGR2BGRA)
        object_img2_trans[:, :, 3] = mask_invert
        crop_img2 = object_img2_trans[y0:y1, x0:x1, :]
        os.makedirs(img_no_mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_no_mask_dir, f"{name}_{str_current_datetime}.png"), crop_img2)

        # save original image
        # cv2.imwrite(os.path.join(crop_dir, f"{crop_idx}_{name}_org.png"), object_img_trans)
        # save information
        # json_data = {
        #     'box': int_box.tolist(),
        #     'scaled_box': [x0, y0, x1, y1]
        # }
        # with open(os.path.join(crop_dir, f"{crop_idx}_box.json"), 'w') as f:
        #     json.dump(json_data, f)

    print(f"Save {crop_idx} crop images")


def process_image(image_path, output_dir,
                  model, transform, specified_tags, tag2text_model,
                  box_threshold, text_threshold, iou_threshold,
                  device):
    # get image name
    # image_name = os.path.basename(image_path).replace('.', '_')
    results_dir = output_dir
    temp_dir = os.path.join(output_dir, f'temp/')
    crop_dir = os.path.join(output_dir, f'crop/')
    # make dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    # load image
    image_pil, image = load_image(image_path)

    # visualize raw image
    image_pil.save(os.path.join(temp_dir, "raw_image.jpg"))

    boxedTag2text_model = tag2text_model.to(device)
    raw_image = image_pil.resize(
        (384, 384))
    raw_image = transform(raw_image).unsqueeze(0).to(device)

    res = inference_tag2text(raw_image, boxedTag2text_model, specified_tags)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    text_prompt = res[0].replace(' |', ',')
    caption = res[2]

    print(f"Caption: {caption}")
    print(f"Tags: {text_prompt}")

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    caption = check_caption(caption, pred_phrases)
    print(f"Revise caption with number: {caption}")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    # avoid no box found error
    if transformed_boxes.nelement() == 0:
        print("No box found")
        return

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    ### draw output image with label ###
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    # plt.title('Tag2Text-Captioning: ' + caption + '\n' + 'Tag2Text-Tagging' + text_prompt + '\n')
    plt.axis('off')
    plt.savefig(
        os.path.join(temp_dir, "automatic_label_output.jpg"),
        bbox_inches="tight", pad_inches=0.0
    )

    ### draw output image without label ###
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

    plt.axis('off')
    plt.savefig(
        os.path.join(temp_dir, "automatic_nolabel_output.jpg"),
        bbox_inches="tight", pad_inches=0.0
    )

    ### draw output no mask image ###
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(
        os.path.join(temp_dir, "no_mask.jpg"),
        bbox_inches="tight", pad_inches=0.0
    )

    ### save mask related image ###
    # save_mask_data(caption, masks, boxes_filt, pred_phrases, results_dir, temp_dir)
    save_mask_crop((H, W), masks, boxes_filt.cpu().numpy(), pred_phrases, temp_dir, results_dir, crop_dir, caption,
                   boxes_filt)

    return caption


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--tag2text_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    tag2text_checkpoint = args.tag2text_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    # ChatGPT or nltk is required when using captions
    # openai.api_key = openai_key
    # if openai_proxy:
    # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize Tag2Text
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(), normalize
    ])
    # filter out attributes and action categories which are difficult to grounding
    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)

    specified_tags = 'None'

    # load model
    tag2text_model = tag2text(pretrained=tag2text_checkpoint,
                              image_size=384,
                              vit='swin_b',
                              delete_tag_index=delete_tag_index)
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tag2text_model.threshold = 0.64
    tag2text_model.eval()

    # get all image in folder only image file
    images_dir = Path(image_path)
    image_extensions = [".jpg", ".jpeg", ".png"]
    files = []
    for file_path in images_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            files.append(file_path)
    print(f"Total {len(files)} images found")
    myDir = str(images_dir).replace("assets/", "")
    # print("myDir", myDir)
    img_details = []
    for idx, file_path in enumerate(files):
        print(f"Processing image: {file_path} [{idx + 1}/{len(files)}]")
        try:
            img_caption = process_image(file_path,
                                        output_dir,
                                        model,
                                        transform,
                                        specified_tags,
                                        tag2text_model,
                                        box_threshold,
                                        text_threshold,
                                        iou_threshold,
                                        device)
            img_details.append({
                'image': str(file_path),
                'caption': img_caption
            })
        except Exception as e:
            print(f"Error: {e}")
            continue
    with open(os.path.join(output_dir, f'{myDir}.json'), 'w') as f:
        json.dump(img_details, f, indent=4)
