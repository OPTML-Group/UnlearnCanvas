# Usage

The experiments in this folder are carried in the environment `unlearn_canvas`. 


## Dataset Preparation

This method needs to forget one concept by replacing one concept with it. Here, we unlearn each style and replace the target style with the photo style.

First, we need to prepare a list of images used for unlearning images. Run this script first:

```bash
python3 generate_dataset.py
```

This does nothing but collect of a lisf of images and its corresponding prompts and store it in the folder `data/`.

## Unlearning

The checkpoint used for SHS is in the compvis format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`compvis/style50/compvis.ckpt`). 

Unlearn command is as follows
```bash
python3 train-erase.py --theme ${theme} --output_dir OUTPUT_DIR --ckpt_path PATH_TO_COMPVIS_CKPT
```

The result will be stored in `args.output_dir/theme/sd.ckpt`.