import os
from PIL import Image
from torchvision import transforms
import torch.utils.data as data


def default_loader(path):
    return Image.open(path).convert('RGB')


def single_load(path, resize=-1):
    img = default_loader(path)
    if resize != -1:
        img = img.resize([resize, resize])
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img_tensor


def load_img_tensor(img_path):
    img = default_loader(img_path)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img_tensor


def get_img_path_list(txt_path):
    img_path_list = []
    f = open(txt_path)
    line = f.readline()
    while line:
        line = line.strip('\n')
        img_path_list.append(line)
        line = f.readline()
    f.close()
    return img_path_list


class Dataset(data.Dataset):
    def __init__(self, args, text_path):
        super(Dataset, self).__init__()
        self.args = args
        self.text_path = text_path
        self.transform = transforms.Compose([
            transforms.Resize(self.args.fineSize),
            transforms.RandomResizedCrop(self.args.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_list = get_img_path_list(self.text_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.args.data_dir, self.img_list[index])
        img = default_loader(img_path)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.img_list)


class TrainDataset(data.Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize(self.args.load_size),
            transforms.RandomResizedCrop(self.args.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_list = get_img_path_list(self.args.train_txt_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.args.data_dir, self.img_list[index])
        img = default_loader(img_path)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.img_list)


class TestDataset(data.Dataset):
    def __init__(self, args):
        super(TestDataset, self).__init__()
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize(self.args.load_size),
            transforms.RandomResizedCrop(self.args.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_list = get_img_path_list(self.args.test_txt_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.args.data_dir, self.img_list[index])
        img = default_loader(img_path)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.img_list)
