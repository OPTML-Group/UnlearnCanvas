from .base_options import BaseOptions


class QCTestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./qc_results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # Set the default = 5000 to test the whole test set.
        parser.add_argument('--num_test', type=int, default=999999999, help='how many test images to run')
        parser.add_argument('--content_img', type=str, required=False, help='content img dir')
        parser.add_argument('--style_img', type=str, required=False, help='style img dir')
        parser.add_argument('--img_dir', type=str, required=True, help='saves result image here.')
        parser.add_argument('--img_name', type=str, required=False, help='saves result image here.')


        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
