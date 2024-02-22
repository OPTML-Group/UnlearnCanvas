import os
from constants.const import theme_available

# Forget classes

for idx, theme in enumerate(theme_available):
    command = f"python3 generate_mask.py --theme {theme} --output_dir results/style50; python3 train-erase.py --theme {theme} --output_dir results/style50/"

    print(command)
    # Run the command
    os.system(command)
