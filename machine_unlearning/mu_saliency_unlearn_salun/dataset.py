import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from PIL import Image
from stable_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from einops import rearrange

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def read_text_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


class CenterSquareCrop:
    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        return F.crop(img, top=int(top), left=int(left), height=min_dim, width=min_dim)


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        CenterSquareCrop(),
        torch_transforms.Resize(size, interpolation=interpolation),
    ])
    return transform


def setup_model(config, ckpt, device):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


# Van Gogh Removal
class StyleDataset(Dataset):
    def __init__(self, data_path, prompt_path, transform=None):
        self.image_paths = read_text_lines(data_path)
        self.prompts = read_text_lines(prompt_path)
        self.transform = transform

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        prompt = self.prompts[idx]

        if self.transform:
            image = self.transform(image)

        # Convert the images to tensors
        image = rearrange(2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w")

        return image, prompt


def setup_forget_style_data(forget_data_dir, remain_data_dir, batch_size, image_size, interpolation='bicubic'):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_data_path = os.path.join(forget_data_dir, 'images.txt')
    forget_prompt_path = os.path.join(forget_data_dir, 'prompts.txt')
    forget_set = StyleDataset(forget_data_path,forget_prompt_path, transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_data_path = os.path.join(remain_data_dir, 'images.txt')
    remain_prompt_path = os.path.join(remain_data_dir, 'prompts.txt')
    remain_set = StyleDataset(remain_data_path, remain_prompt_path, transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl
