# Repeat run different theme for the command: python train.py -t --gpus 0, --concept_type style --caption_target  "Gorgeous Love Style" --prompts ./anchor_prompts/finetune_prompts/painting.txt --name "GorgeousLove"  --train_size 200 --resume_from_checkpoint_custom ../main_sd_image_editing/ckpts/sd_model/lr1e-6/step3999.ckpt
import os
from constants.const import theme_available

for theme in theme_available:
    caption = theme.replace("_", " ")
    command = f"python train.py -t --gpus 0, --concept_type style --caption_target  \"{theme} Style\" --prompts ./anchor_prompts/finetune_prompts/painting.txt --name \"{theme}\" --train_size 200 --resume_from_checkpoint_custom PATH_TO_COMPVIS_CKPT"

    print(command)
    # Run the command
    os.system(command)