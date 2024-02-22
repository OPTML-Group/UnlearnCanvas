code: https://github.com/diyiiyiii/MCCNet
paper: https://arxiv.org/abs/2009.08003
venue: AAAI2021

## Environment

This code is running in the environment `unlearn_canvas`.

## Preparation
Download available ckpts:
vgg-model: https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing
mccnet-model: contained in the repo
decoder: contained in the repo

Move all the checkpoints to `./experiments/`


## Code running

```bash
python3 unlearn_canvas_batch_test.py --output_dir ./eval_results/style_transfer/mcc/style60/ --img_dir PATH_TO_DATASET_DIR
```