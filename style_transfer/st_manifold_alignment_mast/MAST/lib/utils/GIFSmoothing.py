import torch
from PIL import Image
from torch import nn
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
from torchvision import transforms


class GIFSmoothing(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, stylized_tensor, content_tensor):
        """
        :param stylized_tensor: (b,c,h,w)
        :param content_tensor: (b,c,h,w)
        :return:
        """
        # print(f'stylized.size-{stylized_tensor.size()}, content.size={content_tensor.size()}')
        stylized_img = stylized_tensor.clone()
        content_img = content_tensor.clone()
        b, c, h, w = content_img.size()
        device = stylized_img.device
        ori_type = stylized_img.type
        res = []
        for i in range(b):
            s_img = stylized_img[i].float()
            s_img = transforms.ToPILImage()(s_img.cpu()).convert('RGB')
            c_img = content_img[i].float()
            c_img = transforms.ToPILImage()(c_img.cpu()).convert('RGB')
            s_img = s_img.resize((w, h), Image.ANTIALIAS)
            temp = self.process_opencv(s_img, c_img)
            temp = transforms.ToTensor()(temp).to(device).unsqueeze(0).type_as(stylized_tensor)
            res.append(temp.clone())
        res = torch.cat(res, dim=0)
        return res

    def process_opencv(self, initImg, contentImg):
        """
        :param initImg: intermediate output. Either image path or PIL Image
        :param contentImg: content image output. Either path or PIL Image
        :return: stylized output image. PIL Image
        """
        if type(initImg) == str:
            init_img = cv2.imread(initImg)
        else:
            init_img = np.array(initImg)[:, :, ::-1].copy()

        if type(contentImg) == str:
            cont_img = cv2.imread(contentImg)
        else:
            cont_img = np.array(contentImg)[:, :, ::-1].copy()

        output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(output_img)
        return output_img
