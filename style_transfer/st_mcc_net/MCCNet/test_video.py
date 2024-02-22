import argparse
import os
import torch
import torch.nn as nn
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

def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    # return [os.path.join(img_dir,x) for x in files]
    return paths

def load_images(args):
    assert (args.content or args.content_dir)
    assert (args.style or args.style_dir)
    if not args.content:
        content_paths = get_files(content_dir)
    else:
        content_paths = [args.content]
    if not args.style:
        style_paths = get_files(style_dir)
    else:
        style_paths = [args.style]
    return content_paths, style_paths

def load_weights(vgg, decoder, mcc_module):
    vgg.load_state_dict(torch.load(args.vgg_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    mcc_module.load_state_dict(torch.load(args.transform_path))

def test_transform(size, crop):
    transform_list = []
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
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

def image_process(content, style):
    content_tf1 = content_transform()
    content_frame = content_tf1(content)
    #content_frame = torch.tensor(content_frame)

    h, w, c = np.shape(content_frame)
    style_tf1 = style_transform(h, w)
    style = style_tf1(style.convert("RGB"))
    #style = torch.tensor(style)

    if yaml['preserve_color']:
        style = coral(style, content)

    style = style.to(device).unsqueeze(0)
    content = content_frame.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, mcc_module, content, style, alpha)
    output = output.squeeze(0)
    return output.cpu()

#加载视频
def load_video(content_path,style_path, outfile):
    video = cv2.VideoCapture(content_path)

    rate = video.get(5)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得帧宽和帧高
    fps = int(rate)

    video_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.mp4')

    videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                                  (int(width), int(height)))
    return video,videoWriter
def save_frame(output, videoWriter):
    output = output * 255 + 0.5
    output = torch.uint8(torch.clamp(output, 0, 255).permute(1, 2, 0)).numpy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    videoWriter.write(output)  # 写入帧图

def process_video(content_path, style_path, outfile):
    j = 0
    video, videoWriter = load_video(content_path, style_path, outfile)
    while (video.isOpened()):
        j = j + 1
        ret, frame = video.read()
        if not ret:
            break

        if j % 1 == False:
            # 对每一帧进行风格化。
            style = Image.open(style_path)
            content = Image.fromarray(cv2.cvColor(frame, cv2.COLOR_BGR2RGB))
            output = image_process(frame, style)
            # 对风格化后的结果进行额外处理，以存储到视频中
            save_frame(output, videoWriter)

#图像风格化
def process_image(content_path, style_path, outfile):
    image_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.jpg')
    # 对图像进行风格迁移
    content = Image.open(content_path)
    style = Image.open(style_path)
    output = image_process(content, style)
    save_image(output, image_name)


def test(content_paths, style_paths):
    for content_path in content_paths:
        # process one content and one style
        outfile = output_path + '/' + splitext(basename(content_path))[0] + '/'
        if not os.path.exists(outfile):
            os.makedirs(outfile)

        # 视频风格化
        if 'mp4' in content_path:
            for style_path in style_paths:
                process_video(content_path, style_path, outfile)
        # 图像风格化
        else:
            for style_path in style_paths:
                process_image(content_path, style_path, outfile)

def create_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,default="./content/blonde_girl.jpg",
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str,default="./style/candy.jpg",
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')
    parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_160000.pth')
    parser.add_argument('--transform_path', type=str, default='./experiments/mcc_module_iter_160000.pth')
    parser.add_argument('--vgg_path', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--yaml_path', type=str, default='./yaml/test.yaml')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--style_interpolation_weights', type=str, default="")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return args

if __name__ == '__main__':
    args = create_args()
    with open(args.yaml_path,'r') as file :
        yaml =yaml.load(file, Loader=yaml.FullLoader)
    alpha = args.a
    output_path = args.output
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

    content_paths, style_paths = load_images(args)
    test(content_paths, style_paths)



