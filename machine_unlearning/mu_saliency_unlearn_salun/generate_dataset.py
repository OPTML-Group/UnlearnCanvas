from constants.const import theme_available, class_available
import os

if __name__ == "__main__":

    # For style unlearning
    original_data_dir = "../../data/quick-canvas-benchmark"
    new_dir = './data'
    for theme in theme_available:
        os.makedirs(os.path.join(new_dir, theme), exist_ok=True)
        prompt_list = []
        path_list = []
        for class_ in class_available:
            for idx in [1, 2, 3]:
                prompt_list.append(f"A {class_} image in {theme.replace('_', ' ')} style.")
                path_list.append(os.path.join(original_data_dir, theme, class_, f"{idx}.jpg"))
        with open(os.path.join(new_dir, theme, 'prompts.txt'), 'w') as f:
            f.write('\n'.join(prompt_list))
        with open(os.path.join(new_dir, theme, 'images.txt'), 'w') as f:
            f.write('\n'.join(path_list))

    theme = "Seed_Images"
    os.makedirs(os.path.join(new_dir, theme), exist_ok=True)
    prompt_list = []
    path_list = []
    for class_ in class_available:
        for idx in [1, 2, 3]:
            prompt_list.append(f"A {class_} image in Photo style.")
            path_list.append(os.path.join(original_data_dir, theme, class_, f"{idx}.jpg"))
    with open(os.path.join(new_dir, theme, 'prompts.txt'), 'w') as f:
        f.write('\n'.join(prompt_list))
    with open(os.path.join(new_dir, theme, 'images.txt'), 'w') as f:
        f.write('\n'.join(path_list))


    # For class unlearning
    original_data_dir = "../../data/quick-canvas-benchmark"
    new_dir = './data'

    for object_class in class_available:
        os.makedirs(os.path.join(new_dir, object_class), exist_ok=True)
        prompt_list = []
        path_list = []
        for theme in theme_available:
            for idx in [1, 2, 3]:
                prompt_list.append(f"A {object_class} image in {theme.replace('_', ' ')} style.")
                path_list.append(os.path.join(original_data_dir, theme, object_class, f"{idx}.jpg"))
        with open(os.path.join(new_dir, object_class, 'prompts.txt'), 'w') as f:
            f.write('\n'.join(prompt_list))
        with open(os.path.join(new_dir, object_class, 'images.txt'), 'w') as f:
            f.write('\n'.join(path_list))
