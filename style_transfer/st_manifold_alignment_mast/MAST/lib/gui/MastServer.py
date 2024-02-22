# coding=UTF-8
import torch
from torchvision import transforms
from PIL import Image
from lib.models.base_models import Encoder, Decoder
from lib.core.mast import MASTGUI


class MastServer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.cfg.DEVICE == 'cuda' else torch.device(
            'cpu')
        self.mast = MASTGUI(self.cfg)
        self.init_models()

    def init_models(self):

        decoders_path = {
            'r11': self.cfg.TEST.MODEL.DECODER_R11_PATH,
            'r21': self.cfg.TEST.MODEL.DECODER_R21_PATH,
            'r31': self.cfg.TEST.MODEL.DECODER_R31_PATH,
            'r41': self.cfg.TEST.MODEL.DECODER_R41_PATH,
            'r51': self.cfg.TEST.MODEL.DECODER_R51_PATH
        }

        # set the model
        print(f'[Mast]: Loading models...')
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(self.cfg.TEST.MODEL.ENCODER_PATH, map_location=self.device))
        self.encoder = self.encoder.to(self.device)
        print(f'[Mast]: encoder load complete!')
        self.decoders = {}
        for layer_name in self.cfg.TEST.ARTISTIC.LAYERS.split(','):
            decoder = Decoder(layer=layer_name)
            decoder.load_state_dict(torch.load(decoders_path[layer_name], map_location=self.device))
            decoder = decoder.to(self.device)
            self.decoders[layer_name] = decoder
            print(f'[Mast]: decoder {layer_name} load complete!')
        print(f'[Mast]: Load models completely!')

    def process(self, c_path, s_path, c_pos_dict, s_pos_dict, add_mask_type, expand, expand_num):
        """
        :param c_path: style image path
        :param s_path: content image path
        :param c_pos_dict
        :param s_pos_dict
        :param add_mask_type
        :param expand
        :param expand_num
        :return: out_tensor
        """
        img_size = self.cfg.TEST.GUI.IMAGE_SIZE
        c_tensor = transforms.ToTensor()(
            Image.open(c_path).convert('RGB').resize((img_size, img_size))).unsqueeze(0)
        s_tensor = transforms.ToTensor()(
            Image.open(s_path).convert('RGB').resize((img_size, img_size))).unsqueeze(0)
        c_tensor = c_tensor.to(self.device)
        s_tensor = s_tensor.to(self.device)
        print(f'[Mast]: c_tensor.size={c_tensor.size()}, s_tensor.size={s_tensor.size()}')
        style_weight = self.cfg.TEST.ARTISTIC.STYLE_WEIGHT
        with torch.no_grad():
            sf = self.encoder(s_tensor)
        for layer_name in self.cfg.TEST.ARTISTIC.LAYERS.split(','):
            with torch.no_grad():
                print(f'[Mast]: start process {layer_name} ...')
                print(f'[Mast]: encoder...')
                cf = self.encoder(c_tensor)[layer_name]
                print(f'[Mast]: transform...')
                csf = self.mast.transform_with_pos_dict(cf, sf[layer_name], c_pos_dict, s_pos_dict, add_mask_type,
                                                        expand, expand_num)
                csf = style_weight * csf + (1 - style_weight) * cf
                print(f'[Mast]: csf.size={csf.size()}')
                print(f'[Mast]: decoder...')
                out_tensor = self.decoders[layer_name](csf, layer_name)
                print(f'[Mast]: complete! out tensor.size={out_tensor.size()}')
            c_tensor = out_tensor
        return out_tensor
