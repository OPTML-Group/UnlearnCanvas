# coding=UTF-8
import os
import time
import argparse
import torch
from torchvision.utils import save_image
from lib.models.base_models import Encoder, Decoder
from lib.core.mast import MAST
from lib.core.config import get_cfg
from lib.dataset.Loader import single_load


def multi_level_test(encoder, decoders, layers, mast, style_weight, args, device):
    print(f'processing [{args.content_path}] and [{args.style_path}]...')
    c_tensor = single_load(args.content_path, resize=args.resize)
    s_tensor = single_load(args.style_path, resize=args.resize)
    c_tensor = c_tensor.to(device)
    s_tensor = s_tensor.to(device)
    for layer_name in layers:
        with torch.no_grad():
            cf = encoder(c_tensor)[layer_name]
            sf = encoder(s_tensor)[layer_name]
            csf = mast.transform(cf, sf, args.content_seg_path, args.style_seg_path, args.seg_type)
            csf = style_weight * csf + (1 - style_weight) * cf
            out_tensor = decoders[layer_name](csf, layer_name)
        c_tensor = out_tensor
    os.makedirs(args.output_dir, exist_ok=True)
    c_basename = os.path.splitext(os.path.basename(args.content_path))[0]
    s_basename = os.path.splitext(os.path.basename(args.style_path))[0]
    output_path = os.path.join(args.output_dir, f'{c_basename}_{s_basename}.png')
    save_image(out_tensor, output_path, nrow=1, padding=0)
    print(f'[{output_path}] saved...')


def main():
    parser = argparse.ArgumentParser(description='Artistic Test')
    parser.add_argument('--cfg_path', type=str, default='configs/config.yaml',
                        help='config path')
    parser.add_argument('--content_path', '-c', type=str, default='data/default/content/modern.png',
                        help='path of content image')
    parser.add_argument('--style_path', '-s', type=str, default='data/default/style/15.png',
                        help='path of style image')
    parser.add_argument('--content_seg_path', type=str, default=None,
                        help='content_seg_path')
    parser.add_argument('--style_seg_path', type=str, default=None,
                        help='style_seg_path')
    parser.add_argument('--seg_type', type=str, default='dpst',
                        help='the type of segmentation type, [dpst, labelme]')
    parser.add_argument('--resize', type=int, default=-1,
                        help='resize the image, -1: no resize, x: resize to [x, x]')
    parser.add_argument('--output_dir', type=str, default='results/test/default',
                        help='the output dir to save the output image')
    args = parser.parse_args()
    cfg = get_cfg(cfg_path=args.cfg_path)
    assert cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT is False, 'cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT must be False'

    decoders_path = {
        'r11': cfg.TEST.MODEL.DECODER_R11_PATH,
        'r21': cfg.TEST.MODEL.DECODER_R21_PATH,
        'r31': cfg.TEST.MODEL.DECODER_R31_PATH,
        'r41': cfg.TEST.MODEL.DECODER_R41_PATH,
        'r51': cfg.TEST.MODEL.DECODER_R51_PATH
    }

    device = torch.device('cuda') if torch.cuda.is_available() and cfg.DEVICE == 'gpu' else torch.device('cpu')
    # set the model
    print(f'Building models...')
    encoder = Encoder()
    encoder.load_state_dict(torch.load(cfg.TEST.MODEL.ENCODER_PATH, map_location=device))
    encoder = encoder.to(device)
    layers = cfg.TEST.ARTISTIC.LAYERS.split(',')
    decoders = {}
    for layer_name in layers:
        decoder = Decoder(layer=layer_name)
        decoder.load_state_dict(torch.load(decoders_path[layer_name], map_location=device))
        decoder = decoder.to(device)
        decoders[layer_name] = decoder
    print(f'Finish!')

    mast = MAST(cfg)

    style_weight = cfg.TEST.ARTISTIC.STYLE_WEIGHT
    multi_level_test(encoder, decoders, layers, mast, style_weight, args, device)


if __name__ == '__main__':
    main()
