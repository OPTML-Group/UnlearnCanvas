from constants.const import theme_available, class_available
import os

if __name__ == "__main__":
    new_dir = '.'
    os.makedirs(os.path.join(new_dir), exist_ok=True)
    prompt_list = []
    for theme in theme_available:
        for class_object in class_available:
            if theme == "Seed_Images":
                prompt_list.append(f"A {class_object} image in Photo style.")
            else:
            # if theme == "Seed_Images":
            #     continue
                prompt_list.append(f"A {class_object} image in {theme.replace('_', ' ')} style.")
    with open(os.path.join(new_dir, 'fim_prompts.txt'), 'w') as f:
        f.write('\n'.join(prompt_list))
