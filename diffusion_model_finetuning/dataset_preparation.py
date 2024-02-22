from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms
from constants.const import theme_available, class_available

theme_available.remove("Seed_Images")


class CenterSquareCrop:
    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        right = (w + min_dim) / 2
        bottom = (h + min_dim) / 2
        return F.crop(img, top=int(top), left=int(left), height=min_dim, width=min_dim)


class EditDataset(Dataset):
    """
    This is the Dataset for training
    """

    def __init__(
            self,
            path: str,  # "../data/quick-canvas-benchmark"
            split: str = "train",
            splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
            res: int = 256,  # 256
            min_resize_res: int = 256,  # 256
            max_resize_res: int = 512,  # 512
            crop_res: int = 256,  # 256
            flip_prob: float = 0.0,  # 0.5 for train, 0.0 for val and test
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.theme_total = len(theme_available)
        self.class_total = len(class_available)
        self.image_per_class = int(20 * splits[0])
        self.trainable_images = self.theme_total * self.class_total * self.image_per_class
        print(f"Total trainable images: {self.trainable_images}")

        self.transforms = transforms.Compose([
            CenterSquareCrop(),
            transforms.Resize((res, res)),
            transforms.RandomCrop((crop_res, crop_res)),
            transforms.RandomHorizontalFlip(p=flip_prob),
        ])

    def __len__(self) -> int:
        return self.trainable_images

    def __getitem__(self, i: int) -> dict[str, Any]:
        theme_idx = int(i / (self.image_per_class * self.class_total))
        class_idx = int((i % (self.image_per_class * self.class_total)) / self.image_per_class)
        image_idx = int(i % self.image_per_class)
        name = theme_available[theme_idx]

        image_dir = Path(self.path).joinpath(name)
        reference_dir = Path(self.path).joinpath("Seed_Images")

        # The randomly selected images
        image_1 = Image.open(image_dir.joinpath(f"{class_available[class_idx]}/{image_idx + 1}.jpg"))
        image_0 = Image.open(reference_dir.joinpath(f"{class_available[class_idx]}/{image_idx + 1}.jpg"))

        image_0 = self.transforms(image_0)
        image_1 = self.transforms(image_1)

        # Convert the images to tensors
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        prompt = f"Transform the image to {name.replace('_', ' ')} style."

        # return the images and the editing instruction
        # one data item includes:
        #   image_0: the image before editing
        #   image_1: the image after editing
        #   prompt: the editing instruction
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDatasetEval(Dataset):
    """
    This is the dataset for evaluation, the difference lies in that
    the images are not randomly cropped or flipped and only image0
    is returned, which is the to-be-edited image.
    """

    def __init__(
            self,
            path: str,  # "../data/quick-canvas-benchmark"
            split: str = "eval",
            splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
            res: int = 256,  # 256
            crop_res: int = 256,  # 256
            min_resize_res: int = 256,  # 256
            max_resize_res: int = 512,  # 512
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        self.theme_total = len(theme_available)
        self.class_total = len(class_available)
        self.image_per_class = int(20 * (1 - splits[0]))
        self.image_offset = int(20 * splits[0])
        self.trainable_images = self.theme_total * self.class_total * self.image_per_class

        self.transforms = transforms.Compose([
            CenterSquareCrop(),
            transforms.Resize((res, res)),
        ])

    def __len__(self) -> int:
        return self.trainable_images

    def __getitem__(self, i: int) -> dict[str, Any]:
        theme_idx = int(i / (self.image_per_class * self.class_total))
        class_idx = int((i % (self.image_per_class * self.class_total)) / self.image_per_class)
        image_idx = int(i % self.image_per_class)
        name = theme_available[theme_idx]

        image_dir = Path(self.path).joinpath(name)
        reference_dir = Path(self.path).joinpath("Seed_Images")

        # The randomly selected images
        image_1 = Image.open(image_dir.joinpath(f"{class_available[class_idx]}/{image_idx + self.image_offset}.jpg"))
        image_0 = Image.open(reference_dir.joinpath(f"{class_available[class_idx]}/{image_idx + self.image_offset}.jpg"))

        image_0 = self.transforms(image_0)
        image_1 = self.transforms(image_1)

        # Convert the images to tensors
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        prompt = f"Transform the image to {name.replace('_', ' ')} style."

        # return the images and the editing instruction
        # one data item includes:
        #   image_0: the image before editing
        #   image_1: the image after editing
        #   prompt: the editing instruction
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class GenerationDataset(Dataset):
    """
    This is the Dataset for training
    """

    def __init__(
            self,
            path: str,  # "../data/quick-canvas-benchmark"
            split: str = "train",
            splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
            res: int = 256,  # 256
            min_resize_res: int = 256,  # 256
            max_resize_res: int = 512,  # 512
            crop_res: int = 256,  # 256
            flip_prob: float = 0.0,  # 0.5 for train, 0.0 for val and test
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.theme_total = len(theme_available)
        self.class_total = len(class_available)
        self.image_per_class = int(20 * splits[0])
        self.trainable_images = self.theme_total * self.class_total * self.image_per_class
        print(f"Total trainable images: {self.trainable_images}")

        self.transforms = transforms.Compose([
            CenterSquareCrop(),
            transforms.Resize((res, res)),
            transforms.RandomCrop((crop_res, crop_res)),
            transforms.RandomHorizontalFlip(p=flip_prob),
        ])

    def __len__(self) -> int:
        return self.trainable_images

    def __getitem__(self, i: int) -> dict[str, Any]:
        theme_idx = int(i / (self.image_per_class * self.class_total))
        class_idx = int((i % (self.image_per_class * self.class_total)) / self.image_per_class)
        image_idx = int(i % self.image_per_class)
        name = theme_available[theme_idx]

        image_dir = Path(self.path).joinpath(name)

        # The randomly selected images
        image_1 = Image.open(image_dir.joinpath(f"{class_available[class_idx]}/{image_idx + 1}.jpg"))
        image_1 = self.transforms(image_1)

        # Convert the images to tensors
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        if name == "Seed_Images":
            prompt = f"A {class_available[class_idx]} image in Photo style."
        else:
            prompt = f"A {class_available[class_idx]} image in {name.replace('_', ' ')} style."

        return dict(edited=image_1, edit=dict(c_crossattn=prompt))


class GenerationDatasetEval(Dataset):
    """
    This is the dataset for evaluation, the difference lies in that
    the images are not randomly cropped or flipped and only image0
    is returned, which is the to-be-edited image.
    """

    def __init__(
            self,
            path: str,  # "../data/quick-canvas-benchmark"
            split: str = "eval",
            splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
            res: int = 256,  # 256
            crop_res: int = 256,  # 256
            min_resize_res: int = 256,  # 256
            max_resize_res: int = 512,  # 512
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        self.theme_total = len(theme_available)
        self.class_total = len(class_available)
        self.image_per_class = int(20 * (1 - splits[0]))
        self.image_offset = int(20 * splits[0])
        self.trainable_images = self.theme_total * self.class_total * self.image_per_class

        self.transforms = transforms.Compose([
            CenterSquareCrop(),
            transforms.Resize((res, res)),
        ])

    def __len__(self) -> int:
        return self.trainable_images

    def __getitem__(self, i: int) -> dict[str, Any]:
        theme_idx = int(i / (self.image_per_class * self.class_total))
        class_idx = int((i % (self.image_per_class * self.class_total)) / self.image_per_class)
        image_idx = int(i % self.image_per_class)
        name = theme_available[theme_idx]

        image_dir = Path(self.path).joinpath(name)

        # The randomly selected images
        image_1 = Image.open(image_dir.joinpath(f"{class_available[class_idx]}/{image_idx + self.image_offset}.jpg"))
        image_1 = self.transforms(image_1)

        # Convert the images to tensors
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        if name == "Seed_Images":
            prompt = f"A {class_available[class_idx]} image in Photo style."
        else:
            prompt = f"A {class_available[class_idx]} image in {name.replace('_', ' ')} style."

        # return the images and the editing instruction
        # one data item includes:
        #   image_0: the image before editing
        #   image_1: the image after editing
        #   prompt: the editing instruction
        return dict(edited=image_1, edit=dict(c_crossattn=prompt))

import json


def generate_jsonl_for_diffuser(output_file):
    # Pre-defined lists of themes and objects
    theme_available.append("Seed_Images")
    # Open the file in write mode
    with open(output_file, 'w') as file:
        # Traverse through each theme and object
        for theme in theme_available:
            for obj in class_available:
                for index in range(1, 21):  # Index from 1 to 20
                    # Create the file_name string
                    file_name = f"{theme}/{obj}/{index}.jpg"
                    # Create the text string
                    if theme == "Seed_Images":
                        if obj == "Architectures":
                            text = f"An {obj} image in photo style"
                        else:
                            text = f"A {obj} image in photo style"
                    else:
                        if obj == "Architectures":
                            text = f"An {obj} image in {theme.replace('_', ' ')} style"
                        else:
                            text = f"A {obj} image in {theme.replace('_', ' ')} style"
                    # Create the dictionary for the current item
                    item_dict = {
                        "file_name": file_name,
                        "text": text
                    }
                    # Write the JSON object to the file with a newline character to create a JSONL format
                    file.write(json.dumps(item_dict) + '\n')

    print(f"File {output_file} has been created with the specified format.")


if __name__ == '__main__':
    generate_jsonl_for_diffuser("../../data/quick-canvas-benchmark/metadata.jsonl")
