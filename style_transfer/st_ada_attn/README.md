# Usage

code: https://github.com/Huage001/AdaAttN
paper: https://arxiv.org/abs/2108.03647
venue: ICCV2021

## Environment
This code runs in the environment `unlearn_canvas`.

## Preparation

1. Create checkpoints folder `mkdir checkpoints`
2. Download pretrained model from [here](https://drive.google.com/file/d/1XvpD1eI4JeCBIaW5uwMT6ojF_qlzM_lo/view?usp=sharing), move it to checkpoints directory, and unzip:
```bash
mv [Download Directory]/AdaAttN_model.zip checkpoints/
unzip -qq checkpoints/AdaAttN_model.zip
rm checkpoints/AdaAttN_model.zip
```


## Code Running

```bash
python3 unlearn_canvas_batch_test.py --name AdaAttN --model adaattn --load_size 512 --crop_size 512 --image_encoder_path checkpoints/vgg_normalised.pth --gpu_ids 0 --skip_connection_3 --shallow_layer --output_dir ./eval_results/style_transfer/adaattn/style60/ --img_dir PATH_TO_DATASET_DIR
```