import os
import argparse
import torch
from time import time
from torchvision.utils import save_image
from lib.models.base_models import Encoder, Decoder
from lib.core.mast import MAST
from lib.core.config import get_cfg
from lib.dataset.Loader import single_load
from tqdm import tqdm
from constants.const import theme_available, class_available

def multi_level_test(encoder, decoders, layers, mast, content_path, style_path, output_path, style_weight, device, resize=512):
    c_tensor = single_load(content_path, resize=resize)
    s_tensor = single_load(style_path, resize=resize)
    c_tensor = c_tensor.to(device)
    s_tensor = s_tensor.to(device)
    for layer_name in layers:
        with torch.no_grad():
            cf = encoder(c_tensor)[layer_name]
            sf = encoder(s_tensor)[layer_name]
            csf = mast.transform(cf, sf)
            csf = style_weight * csf + (1 - style_weight) * cf
            out_tensor = decoders[layer_name](csf, layer_name)
        c_tensor = out_tensor
    save_image(out_tensor, output_path, nrow=1, padding=0)


def main():
    parser = argparse.ArgumentParser(description='Artistic Test')
    parser.add_argument('--cfg_path', type=str, default='configs/config.yaml',
                        help='config path')
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--resize', type=int, default=512,
                        help='resize the image, -1: no resize, x: resize to [x, x]')
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
    mast = MAST(cfg)
    style_weight = cfg.TEST.ARTISTIC.STYLE_WEIGHT

    for theme in tqdm(theme_available):
        if theme == "Seed_Images":
            continue
        output_dir = os.path.join(args.output_dir, theme)
        os.makedirs(output_dir, exist_ok=True)
        for object_class in class_available:
            for test_img_idx in [19, 20]:
                for style_img_idx in range(1, 19):
                    # Timer starts
                    start_time = time()

                    content_img = os.path.join(args.img_dir, "Seed_Images", object_class, str(test_img_idx) + '.jpg')
                    style_img = os.path.join(args.img_dir, theme, object_class, str(style_img_idx) + '.jpg')
                    output_path = os.path.join(output_dir, f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")

                    multi_level_test(encoder, decoders, layers, mast, content_img, style_img, output_path, style_weight, device)
                    print(f'[{output_path}] saved...')

                    # Timer ends
                    end_time = time()
                    print(f"Time taken: {end_time - start_time: .2f} seconds")


if __name__ == '__main__':
    main()
