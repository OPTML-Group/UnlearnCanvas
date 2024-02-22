# Implementation Notes for ArtFlow

## Summary
code: https://github.com/pkuanjie/ArtFlow
paper: https://arxiv.org/abs/2103.16877 (CVPR 2021)

## Preparation

Download available ckpts:
```
gdown --folder "https://drive.google.com/drive/folders/1w2fHgSBYwjplfeCXI8eOGYpi69CpJBTE"
```

## Environment
This code runs in the environment `unlearn_canvas`.

## Code Running

ArtFlow-AdaIn:
```bash
python3 -u unlearn_canvas_batch_test.py --size 512 --n_flow 8 --n_block 2 --operator wct --decoder experiments/ArtFlow-WCT/glow.pth --output_dir ./eval_results/style_transfer/artflow-wct/style60 --img_dir PATH_TO_DATASET_DIR
```

ArtFlow-WCT:
```bash
python3 -u unlearn_canvas_batch_test.py --size 512 --n_flow 8 --n_block 2 --operator adain --decoder experiments/ArtFlow-AdaIN/glow.pth --output_dir ./eval_results/style_transfer/artflow-adain/style60 --img_dir PATH_TO_DATASET_DIR

```