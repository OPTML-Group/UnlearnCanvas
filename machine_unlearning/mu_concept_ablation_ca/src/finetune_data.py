import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

templates_small = [
    'photo of a {}',
]

templates_small_style = [
    'painting in the style of {}',
]


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True


def get_styled_prompt(caption_target, caption):
    caption = caption.strip()
    r = np.random.choice([0, 1, 2])
    if r == 0:
        if np.random.randint(0, 2) == 0:
            return caption + f', in the style of {caption_target} {caption_target}'
        else:
            return caption + f', in the style of {caption_target}'
    elif r == 1:
        if np.random.randint(0, 2) == 0:
            return f'{caption_target} {caption_target} inspired painting, '+caption
        else:
            return f'{caption_target} inspired painting, '+caption
    elif r == 2:
        if np.random.randint(0, 2) == 0:
            return f'in {caption_target} {caption_target}\'s style, '+caption
        else:
            return f'in {caption_target}\'s style, '+caption


class MaskBase(Dataset):
    def __init__(self,
                 datapath,
                 reg_datapath=None,
                 caption=None,
                 reg_caption=None,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 aug=False,
                 style=True,
                 caption_target=None,
                 repeat=0.
                 ):

        self.aug = aug
        self.repeat = repeat
        self.style = style
        self.caption_target = caption_target
        # multiple style/object fine-tuning
        if self.caption_target is not None and ';' in self.caption_target:
            self.caption_target_list = self.caption_target.split(';')
            self.caption_target = None
        self.templates_small = templates_small
        if self.style:
            self.templates_small = templates_small_style
        print(f"finetune_data Line 68 datapath", datapath)
        if os.path.isdir(datapath):
            self.image_paths1 = [os.path.join(datapath, file_path)
                                 for file_path in os.listdir(datapath) if isimage(file_path)]
        else:
            with open(datapath, "r") as f:
                self.image_paths1 = f.read().splitlines()
                self.image_paths1 = [
                    x.replace(
                        '/sensei-fs/users/nupkumar/',
                        '/grogu/user/nkumari/data_custom_diffusion/final_data/regularization_data/')
                    for x in self.image_paths1]

        self._length1 = len(self.image_paths1)

        self.image_paths2 = []
        self._length2 = 0
        if reg_datapath is not None:
            if os.path.isdir(reg_datapath):
                self.image_paths2 = [os.path.join(reg_datapath, file_path)
                                     for file_path in os.listdir(reg_datapath) if isimage(file_path)]
            else:
                with open(reg_datapath, "r") as f:
                    self.image_paths2 = f.read().splitlines()
                    self.image_paths2 = [
                        x.replace(
                            '/sensei-fs/users/nupkumar/',
                            '/grogu/user/nkumari/data_custom_diffusion/final_data/regularization_data/')
                        for x in self.image_paths2]
            self._length2 = len(self.image_paths2)

        self.labels = {
            "relative_file_path1_": [x for x in self.image_paths1],
            "relative_file_path2_": [x for x in self.image_paths2],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.caption = caption

        if os.path.exists(self.caption):
            self.caption = [x.strip() for x in open(caption, 'r').readlines()]

        self.reg_caption = reg_caption
        if os.path.exists(self.reg_caption):
            self.reg_caption = [x.strip() for x in open(reg_caption, 'r').readlines()]

    def __len__(self):
        if self._length2 > 0:
            return 2*self._length2
        elif self.repeat > 0:
            return self._length1*self.repeat
        else:
            return self._length1

    def __getitem__(self, i):
        # sequentially select one style/object
        if hasattr(self, 'caption_target_list'):
            if self.style or self.caption_target_list[0].startswith('*'):
                self.caption_target = self.caption_target_list[i % len(self.caption_target_list)]
            else:
                self.caption_target = self.caption_target_list[(i % min(self._length1, len(self.caption))) // 1000]

        example = {}

        if i >= self._length2 or self._length2 == 0:
            # print("i >= self._length2 or self._length2 == 0")
            image = Image.open(self.labels["relative_file_path1_"][i % self._length1])
            if isinstance(self.caption, str):
                example["caption"] = np.random.choice(self.templates_small).format(self.caption)
            else:
                example["caption"] = self.caption[i % min(self._length1, len(self.caption))]
            if self.caption_target is not None:
                if self.style:
                    example["caption_target"] = example["caption"]
                    example["caption"] = get_styled_prompt(self.caption_target, example["caption_target"])
                else:
                    example["caption_target"] = example["caption"]
                    general, target = self.caption_target.split('+')
                    if general == '*':
                        example["caption"] = target
                    else:
                        example["caption"] = example["caption_target"].strip().lower().replace(general, target)
        else:
            # print("NONONONON: i >= self._length2 or self._length2 == 0")
            image = Image.open(self.labels["relative_file_path2_"][i % self._length2])
            if isinstance(self.reg_caption, str):
                example["caption"] = np.random.choice(self.templates_small).format(self.reg_caption)
            else:
                example["caption"] = self.reg_caption[i % self._length2]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]

        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = self.flip(image)

        if i > self._length2 or self._length2 == 0:
            if self.aug:
                if np.random.randint(0, 3) < 2:
                    random_scale = np.random.randint(self.size // 3, self.size+1)
                else:
                    random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

                if random_scale % 2 == 1:
                    random_scale += 1
            else:
                random_scale = self.size

            if random_scale < 0.6*self.size:
                add_to_caption = np.random.choice(["a far away ", "very small "])
                example["caption"] = add_to_caption + example["caption"]
                cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

                image = image.resize((random_scale, random_scale), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)

                input_image1 = np.zeros((self.size, self.size, 3), dtype=np.float32)
                input_image1[cx - random_scale // 2: cx + random_scale // 2,
                             cy - random_scale // 2: cy + random_scale // 2, :] = image

                mask = np.zeros((self.size // 8, self.size // 8))
                mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
                     (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.

            elif random_scale > self.size:
                add_to_caption = np.random.choice(["zoomed in ", "close up "])
                example["caption"] = add_to_caption + example["caption"]
                cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
                cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

                image = image.resize((random_scale, random_scale), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)
                input_image1 = image[cx - self.size // 2: cx + self.size //
                                     2, cy - self.size // 2: cy + self.size // 2, :]
                mask = np.ones((self.size // 8, self.size // 8))
            else:
                if self.size is not None:
                    image = image.resize((self.size, self.size), resample=self.interpolation)
                input_image1 = np.array(image).astype(np.uint8)
                input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
                mask = np.ones((self.size // 8, self.size // 8))
        else:
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)
            input_image1 = np.array(image).astype(np.uint8)
            input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))

        # Move the channel dimension of input_image1 from the last to first dimension
        input_image1 = np.moveaxis(input_image1, -1, 0)

        # print("input_image1.shape: ", input_image1.shape)
        # print("mask.shape: ", mask.shape)

        example["image"] = input_image1
        example["mask"] = mask

        return example
