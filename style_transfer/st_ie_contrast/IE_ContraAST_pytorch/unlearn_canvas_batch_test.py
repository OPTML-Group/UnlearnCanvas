import argparse
import os
from time import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import net
from constants.const import theme_available, class_available


def test_transform(img, size):
    transform_list = []
    h, w, _ = np.shape(img)
    if h < w:
        newh = size
        neww = w / h * size
    else:
        neww = size
        newh = h / w * size
    neww = int(neww // 4 * 4)
    newh = int(newh // 4 * 4)
    transform_list.append(transforms.Resize((newh, neww)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)

    parser.add_argument('--steps', type=str, default=1)
    parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='model/decoder_iter_160000.pth')
    parser.add_argument('--transform', type=str, default='model/transformer_iter_160000.pth')

    # Advanced options

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    transform = net.Transform(in_planes=512)
    vgg = net.vgg

    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    transform.load_state_dict(torch.load(args.transform))
    vgg.load_state_dict(torch.load(args.vgg))

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)
    transform.to(device)
    decoder.to(device)

    for theme in tqdm(theme_available):
        if theme == "Seed_Images":
            continue
        output_dir = os.path.join(args.output_dir, theme)
        os.makedirs(output_dir, exist_ok=True)
        for object_class in class_available:
            for test_img_idx in [19, 20]:
                for style_img_idx in range(1, 19):
                    content_img = os.path.join(args.img_dir, "Seed_Images", object_class, str(test_img_idx) + '.jpg')
                    style_img = os.path.join(args.img_dir, theme, object_class, str(style_img_idx) + '.jpg')

                    content = Image.open(content_img).convert('RGB')
                    style = Image.open(style_img).convert('RGB')

                    content_tf = test_transform(content, 512)
                    style_tf = test_transform(style, 512)

                    content = content_tf(content)
                    style = style_tf(style)

                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)

                    # Timer start
                    start = time()
                    with torch.no_grad():
                        for x in range(args.steps):
                            Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                            Content5_1 = enc_5(Content4_1)
                            Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
                            Style5_1 = enc_5(Style4_1)
                            transformed_feature = transform(Content4_1, Style4_1, Content5_1, Style5_1)
                            del Content4_1, Content5_1, Style4_1, Style5_1
                            content = decoder(transformed_feature)
                            content.clamp(0, 255)

                        content = content.cpu()
                        output_path = os.path.join(output_dir,
                                                   f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")
                        save_image(content, output_path)
                        print(f"Saved to {output_path}")
                    # Timer end
                    end = time()
                    print(f"Time elapsed: {end - start:.2f}s")
