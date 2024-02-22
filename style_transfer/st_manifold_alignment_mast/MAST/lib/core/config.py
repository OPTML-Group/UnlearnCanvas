import argparse
from yacs.config import CfgNode as CN

# Configuration variables
cfg = CN()

cfg.DEVICE = 'gpu'

# MAST CORE config
cfg.MAST_CORE = CN()
cfg.MAST_CORE.MAX_USE_NUM = -1
cfg.MAST_CORE.SOFT_LAMBDA = 0.05
cfg.MAST_CORE.K_CROSS = 5
cfg.MAST_CORE.REDUCE_DIM_TYPE = 'avg_pool'
cfg.MAST_CORE.DIM_THRESH = 128
cfg.MAST_CORE.PATCH_SIZE = 1
cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT = False  # False->Artistic, True->PhotoRealistic

# Model
cfg.TEST = CN()
cfg.TEST.MODEL = CN()
cfg.TEST.MODEL.ENCODER_PATH = 'checkpoints/vgg_r51.pth'
cfg.TEST.MODEL.DECODER_R11_PATH = 'checkpoints/Artistic_decoders/dec_r11.pth'
cfg.TEST.MODEL.DECODER_R21_PATH = 'checkpoints/Artistic_decoders/dec_r21.pth'
cfg.TEST.MODEL.DECODER_R31_PATH = 'checkpoints/Artistic_decoders/dec_r31.pth'
cfg.TEST.MODEL.DECODER_R41_PATH = 'checkpoints/Artistic_decoders/dec_r41.pth'
cfg.TEST.MODEL.DECODER_R51_PATH = 'checkpoints/Artistic_decoders/dec_r51.pth'
cfg.TEST.MODEL.SKIP_CONNECTION_DECODER_PATH = 'checkpoints/PhotoRealistic_decoders/decoder_r51_r41_r31.pth'

# TestArtistic
cfg.TEST.ARTISTIC = CN()
cfg.TEST.ARTISTIC.LAYERS = 'r41,r31,r21'
cfg.TEST.ARTISTIC.STYLE_WEIGHT = 0.6

# PhotoRealistic
cfg.TEST.PHOTOREALISTIC = CN()
cfg.TEST.PHOTOREALISTIC.LAYERS = 'r51,r41,r31'
cfg.TEST.PHOTOREALISTIC.POST_SMOOTHING = True
cfg.TEST.PHOTOREALISTIC.FAST_SMOOTHING = True
cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_TYPE = 'AWSC2'
cfg.TEST.PHOTOREALISTIC.SKIP_CONNECTION_WEIGHT = 0.5
cfg.TEST.PHOTOREALISTIC.STYLE_WEIGHT = 1.0

cfg.TEST.GUI = CN()
cfg.TEST.GUI.IMAGE_SIZE = 512
cfg.TEST.GUI.ADD_MASK_TYPE = 'pre'
cfg.TEST.GUI.EXPAND = True
cfg.TEST.GUI.EXPAND_NUM = 20
cfg.TEST.GUI.TEMP_DIR = 'results/temp'


def get_cfg_defaults():
    return cfg.clone()


def update_cfg(cfg_path):
    config = get_cfg_defaults()
    config.merge_from_file(cfg_path)
    return config.clone()


def get_cfg(cfg_path):
    if cfg_path is None:
        return get_cfg_defaults()
    else:
        return update_cfg(cfg_path)
