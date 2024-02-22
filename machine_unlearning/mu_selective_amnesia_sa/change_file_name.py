from constants.const import theme_available, class_available
import os

if __name__ == "__main__":
    image_idx = 0
    for theme in theme_available:
        for class_object in class_available:
            file_name = f"fim_dataset/{theme}_{class_object}_seed188.jpg"
            new_file_name = f"fim_dataset/{image_idx}.jpg"
            os.rename(file_name, new_file_name)
            image_idx += 1