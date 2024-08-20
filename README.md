### Grounded-SAM

Grounded-SAM with RAM or Tag2Text for Automatic Labeling.

Use RAM/Tag2Text to generate tags.
Use Grounded-Segment-Anything to generate the boxes and masks.

### Environment setup

Conda:

```bash
conda create --name GroundedSAM python=3.8
conda activate GroundedSAM
```

### Installation

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
python -m pip install -e GroundingDINO
```

Install RAM & Tag2Text:

```bash
python -m pip install -r ./recognize-anything/requirements.txt
python -m pip install -e ./recognize-anything/
```

### Grounded-SAM with RAM or Tag2Text for Automatic Labeling

**Step 1: Init submodule and download the pretrained checkpoint**

Init submodule:

```bash
cd Grounded-Segment-Anything
git submodule init
git submodule update
```

Download pretrained weights for GroundingDINO, SAM and RAM/Tag2Text:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```

**Step 2: Running the demo with tag2text**

```python
export CUDA_VISIBLE_DEVICES=0
python automatic_label_tag2text_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --tag2text_checkpoint tag2text_swin_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo9.jpg \
  --output_dir "outputs" \
  --box_threshold 0.35 \
  --text_threshold 0.35 \
  --iou_threshold 0.5 \
  --device "cpu"
```
Change device to "cpu" if no cuda 

**Step 3: Running the demo with RAM**

```python
export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo9.jpg \
  --output_dir "outputs" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cpu"
  ```


### TODO

**automatic_label_tag2text_demo_ian.py**

改寫成可以批次處理一個資料夾裡的圖檔。

希望儲存成在img_mask資料夾的圖檔可以crop有圖像的地方。