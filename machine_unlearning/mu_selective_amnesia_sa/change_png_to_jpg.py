# We use this image to change all the .png images in the input folder to .jpg images in the output folder:

import os
from PIL import Image


def convert_png_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Construct full file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename[:-4] + '.jpg')

            # Open the image and convert it to RGB
            image = Image.open(input_path).convert('RGB')

            # Save the image in JPG format
            image.save(output_path, 'JPEG')

            # Delete the old PNG image
            os.remove(input_path)


def main():
    input_folder = 'q_dist/photo_style'  # Replace with your input folder path
    output_folder = 'q_dist/photo_style'  # Replace with your output folder path
    convert_png_to_jpg(input_folder, output_folder)


if __name__ == "__main__":
    main()
