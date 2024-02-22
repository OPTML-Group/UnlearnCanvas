# Usage

This method is based on the environment `ablate`, see `environment.yaml`.


The checkpoint used for CA is in the compvis format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) `compvis/style50/compvis.ckpt`. One example is as follows:

```bash
python train.py -t --gpus 0, --concept_type style --caption_target  "Van Gogh Style" --prompts ./anchor_prompts/finetune_prompts/painting.txt --name "ca_van_gogh"  --train_size 200 --resume_from_checkpoint_custom PATH_TO_COMPVIS_CHECKPOINT
```

Please see `style_script.py` and `class_script.py` for runs for unlearning each style and object class.