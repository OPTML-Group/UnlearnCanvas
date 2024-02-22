import os
from options.qc_test_options import QCTestOptions
from data.base_dataset import get_transform
from models import create_model
import util.util as util
from PIL import Image
from time import time
from tqdm import tqdm

from constants.const import theme_available, class_available

def data_prepare(opt, content_img_dir, style_img_dir):
    content_img = Image.open(content_img_dir).convert('RGB')
    style_img = Image.open(style_img_dir).convert('RGB')
    transform = get_transform(opt)
    transformed_content_img = transform(content_img).unsqueeze(0)
    transformed_style_img = transform(style_img).unsqueeze(0)

    return {'A': transformed_content_img, 'B': transformed_style_img, 'A_paths': content_img_dir, 'B_paths': style_img_dir}


if __name__ == '__main__':
    opt = QCTestOptions().parse()  # get test options

    # In this file, we use opt.results_dir as the directory to save the result image.
    # In this file, we use opt.img_dir as the directory to the source image, including the content image and the style image and the image to be tested.

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.parallelize()
    if opt.eval:
        model.eval()

    for theme in tqdm(theme_available):
        if theme == "Seed_Images":
            continue
        output_dir = os.path.join(opt.results_dir, theme)
        os.makedirs(output_dir, exist_ok=True)
        for object_class in class_available:
            for test_img_idx in [19, 20]:
                for style_img_idx in range(1, 19):
                    # Timer starts
                    start_time = time()
                    content_img = os.path.join(opt.img_dir, "Seed_Images", object_class, str(test_img_idx) + '.jpg')
                    style_img = os.path.join(opt.img_dir, theme, object_class, str(style_img_idx) + '.jpg')
                    data = data_prepare(opt, content_img, style_img)
                    model.set_input(data)  # unpack data from data loader
                    model.test()           # run inference
                    visuals = model.get_current_visuals()  # get image results
                    im = util.tensor2im(visuals['fake_B'])
                    image_pil = Image.fromarray(im)
                    output_path = os.path.join(output_dir, f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")
                    image_pil.save(output_path)
                    print(f"Saved to {output_path}")
                    # Timer ends
                    end_time = time()
                    print(f"Time elapsed: {end_time - start_time:.2f} seconds")


