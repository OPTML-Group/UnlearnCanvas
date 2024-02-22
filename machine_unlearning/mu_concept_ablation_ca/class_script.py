# Repeat run different theme for the command: python train.py -t --gpus 0, --concept_type style --caption_target  "Gorgeous Love Style" --prompts ./anchor_prompts/finetune_prompts/painting.txt --name "GorgeousLove"  --train_size 200 --resume_from_checkpoint_custom ../main_sd_image_editing/ckpts/sd_model/lr1e-6/step3999.ckpt
import os
from constants.const import class_available

for object_class in class_available:
    command = f"python train.py -t --gpus 0, --concept_type object --caption_target  \"An image of {object_class}\" --prompts ./anchor_prompts/finetune_prompts/sd_prompt_{object_class}.txt --name \"{object_class}\" --train_size 80 --resume_from_checkpoint_custom ../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt"

    print(command)
    # Run the command
    os.system(command)


# for class in Architectures Bears Birds Butterfly
# Cats Dogs Fishes Flame
# Flowers Frogs Horses Human
# Jellyfish Rabbits Sandwiches Sea
# Statues Towers Trees Waterfalls;
# do python train.py -t --gpus 0, --concept_type object --caption_target  "An image of ${class}" --prompts ./anchor_prompts/finetune_prompts/sd_prompt_${class}.txt --name "${class}" --train_size 80 --resume_from_checkpoint_custom ../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt; done