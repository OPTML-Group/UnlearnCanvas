# Implementation Notes for CAST

## Summary
code: https://github.com/zyxElsa/CAST_pytorch
paper: http://arxiv.org/abs/2205.09542

## Environment

This code is running in the environment `unlearn_canvas`.

## Preparation

Download available ckpts:
1. The pretrained style classification model is saved at ./models/style_vgg.pth.  
```
gdown "https://drive.google.com/uc?export=download&id=12JKlL6QsVWkz6Dag54K59PAZigFBS6PQ"
```

2. The pretrained content encoder is saved at ./models/vgg_normalised.pth.  
```
gdown "https://drive.google.com/uc?export=download&id=1DKYRWJUKbmrvEba56tuihy1N6VrNZFwl"
```

3. CAST model: the pretrained model is saved at ./checkpoints/CAST_model/*.pth.
```
gdown "https://drive.google.com/uc?export=download&id=11dZqu95QfnAgkzgR1NTJfQutz8JlwRY8"
```
```
unzip CAST_model.zip
```

## Code Running
```bash
python3 unlearn_canvas_batch_test.py --img_dir PATH_TO_DATASET_DIR --results_dir ./eval_results/style_transfer/cast/style60/ --name CAST_model
```