# coding=UTF-8
import os
import time
import argparse
import torch
from torchvision.utils import save_image
from lib.models.base_models import Encoder, DecoderAWSC1, DecoderAWSC2
from lib.core.mast import MAST
from lib.core.config import get_cfg
from lib.dataset.Loader import single_load


def skip_connect_test(encoder, decoder, layers, mast, style_weight, smooth_module, args, device):
    print(f'processing [{args.content_path}] and [{args.style_path}]...')
    c_tensor = single_load(args.content_path, resize=args.resize)
    s_tensor = single_load(args.style_path, resize=args.resize)
    c_tensor = c_tensor.to(device)
    s_tensor = s_tensor.to(device)
    with torch.no_grad():
        cf = encoder(c_tensor)
        sf = encoder(s_tensor)
        csf = {}
        for layer in layers:
            temp = mast.transform(cf[layer], sf[layer], args.content_seg_path, args.style_seg_path, args.seg_type)
            temp = style_weight * temp + (1 - style_weight) * cf[layer]
            csf[layer] = temp
        out_tensor = decoder(csf)
    print(f'Post smoothing start...')
    if smooth_module is not None:
        out_tensor = smooth_module.process(out_tensor, c_tensor)
    print(f'Finish!')
    os.makedirs(args.output_dir, exist_ok=True)
    c_basename = os.path.splitext(os.path.basename(args.content_path))[0]
    s_basename = os.path.splitext(os.path.basename(args.style_path))[0]
    output_path = os.path.join(args.output_dir, f'{c_basename}_{s_basename}.png')
    save_image(out_tensor, output_path, nrow=1, padding=0)
    print(f'[{output_path}] saved...')


def main():
    parser = argparse.ArgumentParser(description='PhotoRealistic Test')
    parser.add_argument('--cfg_path', type=str, default='configs/config.yaml',
                        help='config path')
    parser.add_argument('--content_path', '-c', type=str, default='data/photo_data/content/in1.png',
                        help='path of content image')
    parser.add_argument('--style_path', '-s', type=str, default='data/photo_data/style/tar1.png',
                        help='path of style image')
    parser.add_argument('--content_seg_path', type=str, default=None,
                        help='content_seg_path')
    parser.add_argument('--style_seg_path', type=str, default=None,
                        help='style_seg_path')
    parser.add_argument('--seg_type', type=str, default='dpst',
                        help='the type of segmentation type, [dpst, labelme]')
    parser.add_argument('--resize', type=int, default=-1,
                        help='resize the image, -1: no resize, x: resize to [x, x]')
    parser.add_argument('--output_dir', type=str, default='results/test/photo',
                        help='the output dir to save the output image')
    args = parser.parse_args()
    cfg = get_cfg(cfg_path=args.cfg_path)
    assert cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT is True, 'cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT must be True'

    device = torch.device('cuda') if torch.cuda.is_available() and cfg.DEVICE == 'gpu' else torch.device('cpu')
    # set the model
    print(f'Building models...')
    encoder = Encoder()
    encoder.load_state_dict(torch.load(cfg.TEST.MODEL.ENCODER_PATH))
    encoder = encoder.to(device)
    layers = cfg.TEST.PHOTOREALISTIC.LAYERS.split(',')
    decoder = None
    if cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_TYPE == 'AWSC2':
        decoder = DecoderAWSC2(layers, cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_WEIGHT)
    if cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_TYPE == 'AWSC1':
        decoder = DecoderAWSC1(layers, cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_WEIGHT)
    decoder.load_state_dict(torch.load(cfg.TEST.MODEL.SKIP_CONNECTION_DECODER_PATH, map_location=device))
    decoder = decoder.to(device)
    print(f'Finish!')

    mast = MAST(cfg)

    smooth_module = None
    if cfg.TEST.PHOTOREALISTIC.POST_SMOOTHING:
        if cfg.TEST.PHOTOREALISTIC.FAST_SMOOTHING:
            from lib.utils.GIFSmoothing import GIFSmoothing
            smooth_module = GIFSmoothing(r=35, eps=0.001)
        else:
            from lib.utils.photo_smooth import Propagator
            smooth_module = Propagator()

    style_weight = cfg.TEST.PHOTOREALISTIC.STYLE_WEIGHT
    skip_connect_test(encoder, decoder, layers, mast, style_weight, smooth_module, args, device)


if __name__ == '__main__':
    main()
