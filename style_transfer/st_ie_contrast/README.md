# Implementation Notes for IE ContraAST

## Summary
code: https://github.com/HalbertCH/IEContraAST

paper: https://openreview.net/pdf?id=hm0i-cunzGW  (NeurIPS 2021)

## Preparation
Download available ckpts:
```
gdown "https://drive.google.com/uc?export=download&id=11uddn7sfe8DurHMXa0_tPZkZtYmumRNH"
```
```
unzip model.zip
```

## Environment

This code is running in the environment `unlearn_canvas`.

## Code running
```bash
python3 unlearn_canvas_batch_test.py --output_dir ./eval_results/style_transfer/ie_contrast/style60/ --image_dir PATH_TO_DATASET_DIR
```