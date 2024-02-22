import argparse
import os
import torch
import torch.nn as nn
from time import time
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import net as net
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from constants.const import theme_available, class_available


def load_weights(vgg, decoder, mcc_module):
    vgg.load_state_dict(torch.load(args.vgg_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    mcc_module.load_state_dict(torch.load(args.transform_path))

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


def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))

    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, sa_module, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)

    style_fs, content_f, style_f=feat_extractor(vgg, content, style)
    Fccc = sa_module(content_f,content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc=Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)
    feat = feat * alpha + Fccc * (1 - alpha)
    return decoder(feat)
  
def feat_extractor(vgg, content, style):
  norm = nn.Sequential(*list(vgg.children())[:1])
  enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
  enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
  enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
  enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
  enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

  norm.to(device)
  enc_1.to(device)
  enc_2.to(device)
  enc_4.to(device)
  enc_5.to(device)
  content3_1 = enc_3(enc_2(enc_1(content)))
  Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
  Content5_1 = enc_5(Content4_1)
  Style3_1 = enc_3(enc_2(enc_1(style)))
  Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
  Style5_1 = enc_5(Style4_1)
  

  content_f=[content3_1,Content4_1,Content5_1]
  style_f=[Style3_1,Style4_1,Style5_1]

 
  style_fs = [enc_1(style),enc_2(enc_1(style)),enc_3(enc_2(enc_1(style))),Style4_1, Style5_1]
  
  return style_fs,content_f, style_f


def create_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_160000.pth')
    parser.add_argument('--transform_path', type=str, default='./experiments/mcc_module_iter_160000.pth')
    parser.add_argument('--vgg_path', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--yaml_path', type=str, default=None)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--style_interpolation_weights', type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_args()
    # with open(args.yaml_path,'r') as file :
    #     yaml =yaml.load(file, Loader=yaml.FullLoader)
    alpha = args.a
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    vgg = net.vgg
    network = net.Net(vgg, decoder)
    mcc_module = network.mcc_module
    decoder.eval()
    mcc_module.eval()
    vgg.eval()
    load_weights(vgg, decoder, mcc_module)

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
    mcc_module.to(device)
    decoder.to(device)

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
                    output_path = os.path.join(output_dir,
                                               f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")

                    content = Image.open(content_img).convert('RGB')
                    style = Image.open(style_img).convert('RGB')

                    content_tf = test_transform(content, 512)
                    style_tf = test_transform(style, 512)

                    content = content_tf(content)
                    style = style_tf(style)

                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)

                    with torch.no_grad():
                        output = style_transfer(vgg, decoder, mcc_module, content, style, alpha)
                    output = output.squeeze(0).cpu()

                    save_image(output, output_path)
                    print(f"Saved to {output_path}")

                    end_time = time()
                    print(f"Time taken: {end_time - start_time:.2f} seconds")
