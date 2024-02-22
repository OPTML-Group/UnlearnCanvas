import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F


def image_map(img):
    """
    :param img: the image with only one channel opened by PIL.Image
    :return: an image with the format of PTL.Image
    """
    colormap_dict = {0: (0, 0, 0),
                     1: (128, 0, 0),
                     2: (0, 128, 0),
                     3: (128, 128, 0),
                     4: (0, 0, 128),
                     5: (128, 0, 128),
                     6: (0, 128, 128),
                     7: (128, 128, 128),
                     8: (64, 0, 0),
                     9: (192, 0, 0)}
    img_cat = np.vectorize(colormap_dict.get)(img)

    img_cat1 = np.expand_dims(img_cat[0], axis=2)
    img_cat2 = np.expand_dims(img_cat[1], axis=2)
    img_cat3 = np.expand_dims(img_cat[2], axis=2)
    img = np.concatenate([img_cat1, img_cat2, img_cat3], axis=2)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img


def image_map1(mask):
    colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                         [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                         [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                         [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                         [0, 192, 0], [128, 192, 0], [0, 64, 128]])
    mask = np.array(mask)
    mask_color = colormap[mask].astype(np.uint8)
    mask_color = Image.fromarray(mask_color)
    return mask_color


def whiten_and_color(cF, sF):
    # cF_size=[c, h, w], sF_size=[c, h, w]
    device = cF.device
    cFSize = cF.size()
    cF = cF.view(cFSize[0], -1)
    c_mean = torch.mean(cF, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] * cFSize[2] - 1)
    _, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    sF = sF.view(sFSize[0], -1)
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] * sFSize[2] - 1)
    _, s_e, s_v = torch.svd(styleConv, some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    print(
        f'trace={torch.mm((torch.mm(targetFeature, targetFeature.t()) - torch.mm(sF, sF.t())).t(), (torch.mm(targetFeature, targetFeature.t()) - torch.mm(sF, sF.t()))).trace()}')
    print(f'norm={torch.norm((torch.mm(targetFeature, targetFeature.t()) - torch.mm(sF, sF.t()))) ** 2}')

    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature = targetFeature.view(cFSize[0], cFSize[1], cFSize[2])
    return targetFeature


def batch_split(feature, patch_size, padding=0, stride=1):
    """
    :param feature: size = [n,c,h,w]
    :param patch_size: (3, 3)
    :param padding: 0
    :param stride: 1
    :return: size = [n, c*kernel_size, L]
    """
    if patch_size == (1, 1):
        n, c, h, w = feature.size()
        feature_unfold = feature.view(n, c, -1)
    else:
        feature_unfold = F.unfold(feature, kernel_size=patch_size, padding=padding, stride=stride)
    # print(f'feature_unfold.size = {feature_unfold.size()}')
    return feature_unfold


def batch_concatenate(feature_unfold, origin_size, patch_size, padding=0, stride=1):
    """
    :param feature_unfold: size = [n, c*kernel_size, L]
    :param origin_size: (h, w)
    :param patch_size: (3, 3)
    :param padding: 0
    :param stride: 1
    :return: size = [n, c, h, w]
    """
    if patch_size == (1, 1):
        n, c, h, w = feature_unfold.size()[0], feature_unfold.size()[1], origin_size[0], origin_size[1]
        feature_fold = feature_unfold.view(n, c, h, w)
    else:
        feature_fold = F.fold(feature_unfold, output_size=origin_size, kernel_size=patch_size, padding=padding,
                              stride=stride)
        ones = torch.ones_like(feature_fold)
        ones_unfold = batch_split(ones, patch_size=patch_size)
        ones_fold = F.fold(ones_unfold, output_size=origin_size, kernel_size=patch_size, padding=padding, stride=stride)
        feature_fold = feature_fold / ones_fold
    return feature_fold


def patch_match(cf: torch.Tensor, sf: torch.Tensor, max_use_num):
    """

    :param cf: (C, Hc, Wc)
    :param sf: (C, Hs, Ws)
    :param max_use_num: max_use_num for each pixel
    :return: f (HcWc, HsWs)
    """
    device = cf.device
    cf_size = cf.size()
    sf_size = sf.size()
    cf_n = F.normalize(cf, 2, 0).view(cf_size[0], -1)  # (C, HcWc)
    sf_n = F.normalize(sf, 2, 0).view(sf_size[0], -1)  # (C, HsWs)
    residue_use_num = torch.ones(sf_size[1] * sf_size[2]).type(torch.int).to(device) * max_use_num
    res = torch.zeros(cf_size[1] * cf_size[2], sf_size[1] * sf_size[2]).type_as(cf).to(device)
    sample_list = random.sample(range(cf_size[1] * cf_size[2]), cf_size[1] * cf_size[2])
    for i in sample_list:
        temp = cf_n[:, i].unsqueeze(0)  # (1, C)
        dist = torch.mm(temp, sf_n)  # (1, HsWs)
        max_pos = torch.argmax(dist)
        res[i][max_pos] = 1
        residue_use_num[max_pos] -= 1
        # print(f'dist={dist}')
        # print(f'i={i}, max_pos={max_pos}, dist={dist[0][max_pos]}, '
        #       f'residue_use_num[{max_pos}]={residue_use_num[max_pos]}')
        if residue_use_num[max_pos] == 0:
            sf_n[:, max_pos] = 0
    return res


def patch_match_split(cf_split: torch.Tensor, sf_split: torch.Tensor, max_use_num):
    """

    :param cf_split: (c*kernel_size, L)
    :param sf_split: (c*kernel_size, L)
    :param max_use_num: max_use_num for each pixel
    :return: f (HcWc, HsWs)
    """
    device = cf_split.device
    cf_size = cf_split.size()
    sf_size = sf_split.size()
    cf_n = F.normalize(cf_split, 2, 0)  # (c*kernel_size, L)
    sf_n = F.normalize(sf_split, 2, 0)  # (c*kernel_size, L)
    residue_use_num = torch.ones(sf_size[1]).type(torch.int).to(device) * max_use_num
    res = torch.zeros(cf_size[1], sf_size[1]).type_as(cf_split).to(device)
    sample_list = random.sample(range(cf_size[1]), cf_size[1])
    for i in sample_list:
        temp = cf_n[:, i].unsqueeze(0)  # (1, C)
        dist = torch.mm(temp, sf_n)  # (1, HsWs)
        max_pos = torch.argmax(dist)
        res[i][max_pos] = 1
        residue_use_num[max_pos] -= 1
        # print(f'dist={dist}')
        # print(f'i={i}, max_pos={max_pos}, dist={dist[0][max_pos]}, '
        #       f'residue_use_num[{max_pos}]={residue_use_num[max_pos]}')
        if residue_use_num[max_pos] == 0:
            sf_n[:, max_pos] = 0
    return res


def soft_patch_match_split(cf_split: torch.Tensor, sf_split: torch.Tensor, soft_lambda):
    """

    :param cf_split: (c*kernel_size, L)
    :param sf_split: (c*kernel_size, L)
    :param soft_lambda:
    :return: f (HcWc, HsWs)
    """
    device = cf_split.device
    cf_size = cf_split.size()
    sf_size = sf_split.size()
    cf_n = F.normalize(cf_split, 2, 0)  # (c*kernel_size, L)
    sf_n = F.normalize(sf_split, 2, 0)  # (c*kernel_size, L)
    use_num = torch.zeros(1, sf_size[1]).type_as(cf_split).to(device)
    res = torch.zeros(cf_size[1], sf_size[1]).type_as(cf_split).to(device)
    sample_list = random.sample(range(cf_size[1]), cf_size[1])
    for i in sample_list:
        temp = cf_n[:, i].unsqueeze(0)  # (1, C)
        dist = torch.mm(temp, sf_n)  # (1, HsWs)
        dist -= soft_lambda * use_num
        max_pos = torch.argmax(dist)
        res[i][max_pos] = 1
        use_num[0][max_pos] += 1
        # print(f'dist={dist}')
        # print(f'i={i}, max_pos={max_pos}, dist={dist[0][max_pos]}, '
        #       f'residue_use_num[{max_pos}]={residue_use_num[max_pos]}')
    return res


def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']

    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

    # PIL resize has different order of np.shape
    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR),
                           dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR),
                         dtype=np.float32) / 255.0

    color_content_masks = []
    color_style_masks = []
    for i in range(len(color_codes)):
        color_content_masks.append(torch.from_numpy(_extract_mask(content_seg, color_codes[i])).unsqueeze(0))
        color_style_masks.append(torch.from_numpy(_extract_mask(style_seg, color_codes[i])).unsqueeze(0))
    color_content_masks = torch.cat(color_content_masks, dim=0)
    color_style_masks = torch.cat(color_style_masks, dim=0)
    return color_content_masks, color_style_masks


def load_labelme_seg(content_seg_path, style_seg_path, content_shape, style_shape):
    content_seg = np.asarray(Image.open(content_seg_path).resize(content_shape))
    style_seg = np.asarray(Image.open(style_seg_path).resize(style_shape))
    print(f'content_seg_label={np.unique(content_seg)}， style_seg_label={np.unique(style_seg)}')
    max_color_index = min(np.max(content_seg), np.max(style_seg))
    color_content_masks = []
    color_style_masks = []
    for i in range(max_color_index + 1):
        c_mask_1 = content_seg.copy()
        c_mask_2 = content_seg.copy()
        c_mask_1[c_mask_1 == i] = 1
        c_mask_2[c_mask_2 != i] = 0
        c_mask = c_mask_1.__or__(c_mask_2)
        color_content_masks.append(torch.from_numpy(c_mask).unsqueeze(0))
        s_mask_1 = style_seg.copy()
        s_mask_2 = style_seg.copy()
        s_mask_1[s_mask_1 == i] = 1
        s_mask_2[s_mask_2 != i] = 0
        s_mask = s_mask_1.__or__(s_mask_2)
        color_style_masks.append(torch.from_numpy(s_mask).unsqueeze(0))
    color_content_masks = torch.cat(color_content_masks, dim=0)
    color_style_masks = torch.cat(color_style_masks, dim=0)
    return color_content_masks, color_style_masks


def load_labelme_mask(content_seg_path, style_seg_path, content_shape, style_shape):
    c_mask = np.asarray(Image.open(content_seg_path).resize(content_shape))
    s_mask = np.asarray(Image.open(style_seg_path).resize(style_shape))
    print(f'content_seg_label={np.unique(c_mask)}， style_seg_label={np.unique(s_mask)}')
    max_color_index = min(np.max(c_mask), np.max(s_mask))
    wc, hc = content_shape
    ws, hs = style_shape
    c_mask_tensor = torch.from_numpy(c_mask)
    s_mask_tensor = torch.from_numpy(s_mask)
    mask = torch.zeros(hc * wc, hs * ws).int()
    for color_index in range(max_color_index + 1):
        c_pos = torch.where(c_mask_tensor == color_index)
        s_pos = torch.where(s_mask_tensor == color_index)
        c_index = c_pos[0] * hc + c_pos[1]
        s_index = s_pos[0] * hs + s_pos[1]
        col = torch.zeros(hc * wc, 1).int()
        col.index_fill_(0, c_index, 1)
        row = torch.zeros(1, hs * ws).int()
        row.index_fill_(1, s_index, 1)
        temp = torch.mm(col, row)
        mask = mask.__or__(temp)
    return mask


def adjust_learning_rate(optimizer, iteration, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / (1 + iteration * args.lr_decay_rate)


def adjust_tv_loss_weight(args, iteration, ori_tv_loss_weight):
    args.tv_loss_weight = ori_tv_loss_weight / (1 + iteration * args.tv_loss_weight_decay_rate)


def main():
    ori_tv_weight = 1e-6
    for iteration in range(1, 19572 * 4 + 1):
        print(f'iteration={iteration}, weight={ori_tv_weight / (1 + iteration * 1e-3)}')


if __name__ == '__main__':
    main()
