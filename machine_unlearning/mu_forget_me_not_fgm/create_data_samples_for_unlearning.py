from constants.const import theme_available, class_available
import os

dataset_dir = '../../data/quick-canvas-benchmark'
# check if data folder exists and if not, create data folder
if not os.path.exists('data'):
    os.mkdir('data')

for theme in theme_available:
    if not os.path.exists(f'data/{theme}'):
        mkdir_command = f"mkdir data/{theme}"
        os.system(mkdir_command)
    for i, object_class in enumerate(class_available):
        cp_command = f"cp {dataset_dir}/{theme}/{object_class}/1.jpg data/{theme}/{i}.jpg"
        os.system(cp_command)

for object_class in class_available:
    for i, theme in enumerate(theme_available):
        cp_command = f"cp {dataset_dir}/{theme}/{object_class}/1.jpg data/{theme}/{i}.jpg"
        os.system(cp_command)


