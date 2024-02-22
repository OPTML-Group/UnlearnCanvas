import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

import os
import glob
from PIL import Image
from pathlib import Path
from einops import rearrange
import numpy as np


class VisualizationDataset(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images for visualizing while training"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]
    

class TextImageDataset(Dataset):
    
    def __init__(self, caption, img_dir, image_key='image', txt_key='txt'):
        """
        Args:
            caption (string): String of comma-separated captions.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.img_dir = img_dir
        self.all_imgs = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        self.captions = [item.strip() for item in caption.split(",")]
        self.image_key = image_key
        self.txt_key = txt_key
        self.transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.img_dir, str(idx)+ ".jpg")
        image = Image.open(img_name)
        text_cond =  np.random.choice(self.captions)
        image = self.transform(image).permute(1,2,0) # [HxWxC]
        
        return {self.image_key: image, self.txt_key: text_cond}
        

class FileImageDataset(Dataset):
    
    def __init__(self, caption_dir, img_dir, image_key='image', txt_key='txt'):
        """
        Args:
            caption_dir (string): Path to file with captions.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.img_dir = img_dir
        self.all_imgs = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        
        with open(caption_dir, "r") as f:
            data = f.read().splitlines()
            self.captions = list(data)

        print("Number of images: ", len(self.all_imgs))
        print("Number of captions: ", len(self.captions))
        assert len(self.captions) == len(self.all_imgs)
        self.image_key = image_key
        self.txt_key = txt_key
        self.transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.img_dir, str(idx)+ ".jpg")
        image = Image.open(img_name)
        text_cond = self.captions[idx]
        image = self.transform(image).permute(1,2,0) # [HxWxC]
        
        return {self.image_key: image, self.txt_key: text_cond}


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    

def LoadFIMDataLoader(prompt_file, img_dir):
    
    dataset = FileImageDataset(prompt_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    return dataloader


def ForgettingDataset(forget_prompt, forget_dataset_path,
                      replay_prompt_path="fim_prompts.txt", 
                      replay_dataset_path="fim_dataset"):

    concat_dataset = ConcatDataset(
        TextImageDataset(forget_prompt, forget_dataset_path),
        FileImageDataset(replay_prompt_path, replay_dataset_path)
    )
    
    return concat_dataset
