import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.StyTR as StyTR
import models.transformer as transformer
from tqdm import tqdm
from constants.const import theme_available, class_available
from time import time

def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transform(h, w):
    transform_list = []
    transform_list.append(transforms.CenterCrop((h, w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--img_dir', default="../../../data/quick-canvas-benchmark/", type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
    parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
    parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')

    parser.add_argument('--style_interpolation_weights', type=str, default="")
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    args = parser.parse_args()

    # Advanced options
    content_size = 512
    style_size = 512
    crop = 'store_true'
    save_ext = '.jpg'
    preserve_color = 'store_true'
    alpha = args.a

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        # namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.Trans_path)
    for k, v in state_dict.items():
        # namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.embedding_path)
    for k, v in state_dict.items():
        # namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.eval()
    network.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

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

                    content = content_tf(Image.open(content_img).convert("RGB"))

                    h, w, c = np.shape(content)
                    style_tf1 = style_transform(h, w)
                    style = style_tf(Image.open(style_img).convert("RGB"))

                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)

                    with torch.no_grad():
                        output = network(content, style)
                    output = output[0].cpu()
                    output_path = os.path.join(output_dir, f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")
                    save_image(output, output_path)

                    end_time = time()
                    print(f"Time elapse: {end_time - start_time:.2f}s")
