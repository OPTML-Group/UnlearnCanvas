## Resource 

Code: https://github.com/GlebSBrykin/SANET
Paper: https://arxiv.org/abs/1812.02342v5
Venue: CVPR 2019

## Preparation

Download the checkpoints from:

1. [decoder](https://yadi.sk/d/xsZ7j6FhK1dmfQ)
2. [Transformer](https://yadi.sk/d/GhQe3g_iRzLKMQ)
3. [vgg_normalised](https://yadi.sk/d/7IrysY8q8dtneQ)

and move them to the `checkpoints` directory.

## Environment

This code is running in the environment `unlearn_canvas`.

## Code Running

```bash
python3 unlearn_canvas_batch_test.py --output_dir ./eval_results/style_transfer/sanet/style60/ --img_dir PATH_TO_DATASET_DIR
```