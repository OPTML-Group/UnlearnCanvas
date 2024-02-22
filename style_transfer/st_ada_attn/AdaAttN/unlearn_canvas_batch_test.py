import os

from PIL import Image
from tqdm import tqdm
from time import time
import util.util as util
from constants.const import theme_available, class_available
from data.base_dataset import get_transform
from models import create_model
from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--img_dir', type=str, required=True,
                            help='path to the directory of the source image.')
        parser.add_argument('--output_dir', type=str, required=True, help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    for theme in tqdm(theme_available):
        if theme == "Seed_Images":
            continue
        output_dir = os.path.join(opt.output_dir, theme)
        os.makedirs(output_dir, exist_ok=True)
        for object_class in class_available:
            for test_img_idx in [19, 20]:
                for style_img_idx in range(1, 19):
                    start_time = time()
                    content_img = os.path.join(opt.img_dir, "Seed_Images", object_class, str(test_img_idx) + '.jpg')
                    style_img = os.path.join(opt.img_dir, theme, object_class, str(style_img_idx) + '.jpg')
                    output_path = os.path.join(output_dir, f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")

                    A_img = Image.open(content_img).convert('RGB')
                    B_img = Image.open(style_img).convert('RGB')

                    transform = get_transform(opt)
                    A = transform(A_img).unsqueeze(0)
                    B = transform(B_img).unsqueeze(0)

                    data = {'c': B, 's': A, 'name': output_path}

                    model.set_input(data)  # unpack data from data loader
                    model.test()  # run inference
                    visuals = model.get_current_visuals()  # get image results
                    im = util.tensor2im(visuals['s'])
                    image_pil = Image.fromarray(im)
                    image_pil.save(output_path)
                    print(f"Saved to {output_path}")

                    print(f"Time taken: {time() - start_time:.2f}s")
