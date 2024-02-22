import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from constants.const import theme_available, class_available
from time import time

def test_transform(img, size):
    transform_list = []
    h, w, _ = np.shape(img)
    if h<w:
        newh = size
        neww = w/h*size
    else:
        neww = size
        newh = h/w*size
    neww = int(neww//4*4)
    newh = int(newh//4*4)
    transform_list.append(transforms.Resize((newh, neww)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--img_dir', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)

parser.add_argument('--decoder', type=str, default='experiments/decoder2.pth.tar')

# Additional options
parser.add_argument('--size', type=int, default=512,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')


# glow parameters
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.operator == 'wct':
    from glow_wct import Glow
elif args.operator == 'adain':
    from glow_adain import Glow
elif args.operator == 'decorator':
    from glow_decorator import Glow
else:
    raise('Not implemented operator', args.operator)

os.makedirs(args.img_dir, exist_ok=True)

# glow
glow = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# -----------------------resume training------------------------
if os.path.isfile(args.decoder):
    print("--------loading checkpoint----------")
    checkpoint = torch.load(args.decoder)
    args.start_iter = checkpoint['iter']
    glow.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.decoder))
else:
    print("--------no checkpoint found---------")
glow = glow.to(device)

glow.eval()

# -----------------------start------------------------

for theme in tqdm(theme_available):
    if theme == "Seed_Images":
        continue
    output_dir = os.path.join(args.output_dir, theme)
    os.makedirs(output_dir, exist_ok=True)
    for object_class in class_available:
        for test_img_idx in [19, 20]:
            for style_img_idx in range(1, 19):
                start_time = time()
                content_img = os.path.join(args.img_dir, "Seed_Images", object_class, str(test_img_idx) + '.jpg')
                style_img = os.path.join(args.img_dir, theme, object_class, str(style_img_idx) + '.jpg')

                with torch.no_grad():
                    content = Image.open(content_img).convert('RGB')
                    img_transform = test_transform(content, args.size)
                    content = img_transform(content)
                    content = content.to(device).unsqueeze(0)

                    style = Image.open(style_img).convert('RGB')
                    img_transform = test_transform(style, args.size)
                    style = img_transform(style)
                    style = style.to(device).unsqueeze(0)

                    # content/style ---> z ---> stylized
                    z_c = glow(content, forward=True)
                    z_s = glow(style, forward=True)
                    output = glow(z_c, forward=False, style=z_s)
                    output = output.cpu()
                    output_path = os.path.join(output_dir, f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")
                    save_image(output, output_path)
                    print(f"Saved to {output_path}")

                print(f"Time elapsed: {time() - start_time:.2f}s")


            
