# Usage

The experiments in this folder are carried in the environment `unlearn_canvas`. 

This method needs some images for each to-be-unlearned concept. Therefore, we need to prepare a folder containing a subset of the image for each style. For each style, I use the script `create_data_samples_for_unlearning.py` to select `1.jpg` of each class and copy them to the corresponding `data/style/` folder with a renamed file name from `1.jpg` to `20.jpg` (there are 20 classes in each style) as the unlearning images.


## Unlearning

The checkpoint used for FMN is in the diffuser format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`diffuser/style50`). 

This method involves two stages. The first one is to train a Text Inversion, see `train_ti.py` and then use the outputs from the stage to perform unlearning, see `train_attn.py`. 

The following scripts are used to perform unlearning as an example:

```bash
python3 train_ti.py --pretrained_path PATH_TO_DIFFUSER_DIR --theme Abstractionism --output_dir OUTPUT_DIR --steps 500 --lr 1e-4
```

```bash
python3 train_attn.py --theme Abstractionism --lr 2e-5 --max-steps 100 --ti_weight_path OUTPUT_DIR/Abstractionism/step_inv_500.safetensors --output_dir OUTPUT_DIR --only_xa
```

When using `train_attn.py`, there are three parameters that can be tuned: 1. `--lr`, 2. `--max-steps`, 3. `--only_xa`. Generally speaking, a large lr value and a large step number will cause the model to unlearn better, but it will also easily cause the model to corrupt (can not generate anything). Usually, turn on `--only-xa` can improve the stability of the model.

For style unlearning, a relatively good choice is `--lr=2e-5 --max-steps 100 --only_xa`.

For topic unlearning, a relatively good choice is `--lr=2e-5 --max-steps 100`.