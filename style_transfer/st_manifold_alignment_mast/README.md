## Resources

Code: https://github.com/NJUHuoJing/MAST
Paper: https://arxiv.org/pdf/2005.10777.pdf
Venue: ICCV 2021

## Preparation

Download the checkpoint from: https://drive.google.com/file/d/16R7monpAEN_hFuQPvaB6sYuBJMyaJiK7/view?usp=sharing
put the checkpoints under: `./st_manifold_alignment_mast/MAST/`
unzip the checkpoints: `unzip -qq checkpoints.zip`

## Environment

This code is running in the environment `unlearn_canvas`.

## Coding Running

```bash
python3 unlearn_canvas_batch_test.py --output_dir ./eval_results/style_transfer/mast/style60/ --image_dir PATH_TO_DATASET_DIR
```