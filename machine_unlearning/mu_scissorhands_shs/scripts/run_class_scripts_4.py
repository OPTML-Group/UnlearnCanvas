import os
import sys
sys.path.append(".")
from constants.const import class_available

# Forget classes

remain_class_available = ['Bears', 'Birds', 'Butterfly', 'Cats', 'Dogs', 'Fishes', 'Flame', 'Flowers', 'Frogs', 'Horses', 'Human', 'Jellyfish', 'Rabbits', 'Sandwiches', 'Sea', 'Statues', 'Towers', 'Trees', 'Waterfalls', 'Architectures']

class_available = class_available[7:10]
remain_class_available = remain_class_available[7:10]

for idx, object_class in enumerate(class_available):
    command = f"python3 generate_mask.py --theme {object_class} --output_dir results/style50; python3 train-erase.py --theme {object_class} --output_dir results/style50/ --remain_data_dir data/{remain_class_available[idx]}"

    print(command)
    # Run the command
    os.system(command)
