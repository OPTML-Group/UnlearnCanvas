import os
import shutil
import argparse

from constants.const import theme_available, class_available

def copy_config_file(config_file, target_file, key, target):
    # Check if the source config file exists
    if not os.path.exists(config_file):
        print(f"The config file {config_file} does not exist.")
        return

    # Copy the file
    shutil.copy(config_file, target_file)

    # Read from the copied file and replace the key
    with open(target_file, 'r') as file:
        lines = file.readlines()

    with open(target_file, 'w') as file:
        for line in lines:
            file.write(line.replace(key, target))

def main():
    parser = argparse.ArgumentParser(description="Copy a config file and replace a specified keyword.")
    parser.add_argument("--config-file", default="configs/generate_sd.yaml", type=str, help="Path to the source config file")
    parser.add_argument("--target-file", default="configs/generate_sd.yaml", type=str, help="Path to the target config file")
    parser.add_argument("--key", default="KEY", type=str, help="Keyword to be replaced")
    parser.add_argument("--target", default="TARGET", type=str, help="Replacement for the keyword")
    args = parser.parse_args()

    copy_config_file(args.config_file, args.target_file, args.key, args.target)

if __name__ == "__main__":
    theme_available.remove("Abstractionism")
    for theme in theme_available:
        config_file = f"configs/forget_Abstractionism.yaml"
        target_file = f"configs/forget_{theme}.yaml"
        copy_config_file(config_file, target_file, "Abstractionism", theme)
