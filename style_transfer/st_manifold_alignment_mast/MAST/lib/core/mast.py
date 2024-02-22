import torch
import gc
import os
import cv2 as cv
import numpy as np
import torch.nn.functional as f
from lib.utils.utils import batch_split, batch_concatenate, load_seg, load_labelme_mask
from lib.utils.utils import patch_match_split, soft_patch_match_split, image_map1


class MAST(object):
    def __init__(self, cfg):
        """
        the type of all values are the same as cf
        :param cfg: config
        """
        self.cfg = cfg
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.cfg.DEVICE == 'gpu' else torch.device(
            'cpu')
        if cfg.MAST_CORE.REDUCE_DIM_TYPE == 'avg_pool':
            self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif cfg.MAST_CORE.REDUCE_DIM_TYPE == 'max_pool':
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def cal_p(self, cf, sf, mask=None):
        """
        :param cf: [c*kernel_size, hcwc]
        :param sf: [c*kernel_size, hsws]
        :param mask: [hcwc, hsws]
        :return:p [c*c]
        """
        cf_size = cf.size()
        sf_size = sf.size()
        k_cross = self.cfg.MAST_CORE.K_CROSS

        cf_temp = cf.clone()
        sf_temp = sf.clone()

        if self.cfg.MAST_CORE.MAX_USE_NUM == -1:
            # ########################################
            # normalize
            cf_n = f.normalize(cf, 2, 0)
            sf_n = f.normalize(sf, 2, 0)
            # #########################################

            dist = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar
            if mask is not None:
                mask = mask.type_as(dist).to(self.device)
                dist = torch.mul(dist, mask)

            hcwc, hsws = cf_size[1], sf_size[1]
            U = torch.zeros(hcwc, hsws).type_as(cf_n).to(self.device)  # construct affinity matrix "(h*w)*(h*w)"

            index = torch.topk(dist, k_cross, 0)[1]  # find indices k nearest neighbors along row dimension
            value = torch.ones(k_cross, hsws).type_as(cf_n).to(self.device) # "KCross*(h*w)"
            U.scatter_(0, index, value)  # set weight matrix
            del index
            del value
            gc.collect()

            index = torch.topk(dist, k_cross, 1)[1]  # find indices k nearest neighbors along col dimension
            value = torch.ones(hcwc, k_cross).type_as(cf_n).to(self.device)
            U.scatter_(1, index, value)  # set weight matrix
            del index
            del value
            gc.collect()
        elif self.cfg.MAST_CORE.MAX_USE_NUM == 0:
            U = soft_patch_match_split(cf, sf, soft_lambda=self.cfg.MAST_CORE.SOFT_LAMBDA)
        else:
            U = patch_match_split(cf, sf, max_use_num=self.cfg.MAST_CORE.MAX_USE_NUM)
        n_cs = torch.sum(U)
        U = U / n_cs
        D1 = torch.diag(torch.sum(U, dim=1)).type_as(cf).to(self.device)
        if self.cfg.MAST_CORE.ORTHOGONAL_CONSTRAINT:
            A = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
            # regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.device) * 1e-12
            # A += regularization_term
            A_U, A_S, A_V = torch.svd(A)
            p = torch.mm(A_U, A_V.t())
        else:
            try:
                A = torch.mm(torch.mm(cf_temp, D1), cf_temp.t())
                regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.device) * 1e-12
                A += regularization_term
                B = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
                p = torch.solve(B, A).solution
            except Exception as e:
                print(e)
                p = torch.eye(cf_size[0]).type_as(cf).to(self.device)
        return p

    def cal_csf(self, ori_cf, cf, sf, mask=None):
        """
        :param ori_cf:
        :param cf: [n, c*kernel_size, hcwc]
        :param sf: [n, c*kernel_size, hsws]
        :param mask: [hcwc, hsws]
        :return: csf [n, c*kernel_size, hcwc]
        """
        cf_size = cf.size()
        sf_size = sf.size()
        if cf_size[0] != sf_size[0] or cf_size[1] != sf_size[1]:
            csf = cf
        else:
            csf = []
            for i in range(cf_size[0]):
                ori_cf_temp = ori_cf[i]
                cf_temp = cf[i]
                sf_temp = sf[i]
                p = self.cal_p(cf_temp, sf_temp, mask)
                csf_temp = torch.mm(p.t(), ori_cf_temp).unsqueeze(0)
                csf.append(csf_temp)
            csf = torch.cat(csf, dim=0)
        return csf

    def can_seg(self, content_seg_path, style_seg_path):
        if self.cfg.MAST_CORE.PATCH_SIZE != 1:
            print(f'patch size = {self.cfg.MAST_CORE.PATCH_SIZE}, must be 1, can not use segmentation...')
            return False
        if self.cfg.MAST_CORE.MAX_USE_NUM != -1:
            print(f'max use num={self.cfg.MAST_CORE.MAX_USE_NUM}, must be -1, can not use segmentation...')
            return False
        if not os.path.exists(content_seg_path):
            print(f'content segmentation image [{content_seg_path}] not exists...')
            return False
        if not os.path.exists(style_seg_path):
            print(f'style segmentation image [{style_seg_path}] not exists...')
            return False
        return True

    def down_sampling_feature(self, cf, sf):
        thresh = self.cfg.MAST_CORE.DIM_THRESH * self.cfg.MAST_CORE.DIM_THRESH
        while cf.size()[2] * cf.size()[3] > thresh:
            cf = self.pool(cf)
        while sf.size()[2] * sf.size()[3] > thresh:
            sf = self.pool(sf)
        return cf, sf

    def transform(self, cf, sf, content_seg_path=None, style_seg_path=None, seg_type='dpst'):
        """
        :param cf: [n, c, hc, wc]
        :param sf: [n, c, hs, ws]
        :param content_seg_path: content segmentation path
        :param style_seg_path: style segmentation path
        :param seg_type: segmentation type
        :return: csf [n, c, hc, wc]
        """
        ori_cf = cf.clone()
        ori_sf = sf.clone()
        ori_cf_size = ori_cf.size()
        ori_sf_size = ori_sf.size()
        cf, sf = self.down_sampling_feature(cf, sf)
        cf_size = cf.size()
        sf_size = sf.size()

        hc, wc = cf_size[2], cf_size[3]
        hs, ws = sf_size[2], sf_size[3]
        mask = None
        if content_seg_path is not None and style_seg_path is not None:
            if not self.can_seg(content_seg_path, style_seg_path):
                return cf
            if seg_type == 'dpst':
                c_masks, s_masks = load_seg(content_seg_path, style_seg_path, (wc, hc), (ws, hs))
                c_masks = c_masks.view(c_masks.size()[0], -1)
                s_masks = s_masks.view(s_masks.size()[0], -1)
                c_masks = c_masks.unsqueeze(2)
                s_masks = s_masks.unsqueeze(1)
                masks = torch.bmm(c_masks, s_masks)
                mask = masks.max(dim=0).values
                mask = mask.to(self.device)
            elif seg_type == 'labelme':
                mask = load_labelme_mask(content_seg_path, style_seg_path, (wc, hc), (ws, hs))
        patch_size = self.cfg.MAST_CORE.PATCH_SIZE
        cf_split = batch_split(cf, patch_size=(patch_size, patch_size))
        sf_split = batch_split(sf, patch_size=(patch_size, patch_size))
        ori_cf_split = batch_split(ori_cf, patch_size=(patch_size, patch_size))
        csf = self.cal_csf(ori_cf_split, cf_split, sf_split, mask)
        csf = batch_concatenate(csf, origin_size=(ori_cf_size[2], ori_cf_size[3]),
                                patch_size=(patch_size, patch_size))
        return csf


class MASTGUI(MAST):
    def __init__(self, cfg):
        super(MASTGUI, self).__init__(cfg=cfg)

    def can_use_mask(self):
        if self.cfg.MAST_CORE.PATCH_SIZE != 1:
            print(f'patch size = {self.cfg.MAST_CORE.PATCH_SIZE}, must be 1, can not use mask...')
            return False
        return True

    @staticmethod
    def down_sample_pos_set(c_pos_set, s_pos_set, c_ratio, s_ratio):
        new_c_pos_set = set()
        new_s_pos_set = set()
        for c_pos in c_pos_set:
            new_c_pos = int(c_pos / c_ratio)
            new_c_pos_set.add(new_c_pos)
        for s_pos in s_pos_set:
            new_s_pos = int(s_pos / s_ratio)
            new_s_pos_set.add(new_s_pos)
        return new_c_pos_set, new_s_pos_set

    def cal_mask(self, c_mask, s_mask, cf_size, sf_size, add_mask_type, flag):
        """
        :param c_mask: np.array, type=np.uint8
        :param s_mask: np.array, type=np.uint8
        :param cf_size:
        :param sf_size:
        :param add_mask_type
        :param flag:
        :return: torch.tensor type=torch.int(), size=(hc*wc, hs*ws)
        """
        hc, wc = cf_size[2], cf_size[3]
        hs, ws = sf_size[2], sf_size[3]
        c_mask = cv.resize(c_mask, (hc, wc), cv.INTER_NEAREST)
        s_mask = cv.resize(s_mask, (hs, ws), cv.INTER_NEAREST)
        if flag:
            c_mask_img = image_map1(c_mask)
            s_mask_img = image_map1(s_mask)
            c_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 'c_mask_resized.png'))
            s_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 's_mask_resized.png'))
        max_color_index = np.max(c_mask)
        c_mask_tensor = torch.from_numpy(c_mask)
        s_mask_tensor = torch.from_numpy(s_mask)
        mask = torch.zeros(hc * wc, hs * ws).int()
        if add_mask_type == 'pre':
            start_color_index = 0
        else:
            start_color_index = 1
        for color_index in range(start_color_index, max_color_index + 1):
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

    def cal_mask_with_expand(self, c_pos_dict, s_pos_dict, cf, sf, add_mask_type, expand_num):
        c_mask, s_mask = self.cal_c_and_s_mask(c_pos_dict, s_pos_dict)
        c_mask_img = image_map1(c_mask)
        s_mask_img = image_map1(s_mask)

        c_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 'c_mask_origin.png'))
        s_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 's_mask_origin.png'))

        print(f'[Mast]: expand c pos dict...')
        print(f'[Mast]: expand num={expand_num}')
        cf_size = cf.size()
        sf_size = sf.size()
        hc, wc = cf_size[2], cf_size[3]
        hs, ws = sf_size[2], sf_size[3]
        c_mask = cv.resize(c_mask, (hc, wc), cv.INTER_NEAREST)
        s_mask = cv.resize(s_mask, (hs, ws), cv.INTER_NEAREST)
        c_mask_img = image_map1(c_mask)
        s_mask_img = image_map1(s_mask)
        c_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 'c_mask_resized.png'))
        s_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 's_mask_resized.png'))

        # cf = cf.cpu()
        # sf = sf.cpu()
        cf_n = f.normalize(cf[0].view(cf_size[1], -1), 2, 0)
        print(f'cf_n.size={cf_n.size()}, cf_n.device={cf_n.device}')
        # cf_n = cf_n.to('cpu')
        print(f'cf_n.size={cf_n.size()}, cf_n.device={cf_n.device}')
        dist = torch.mm(cf_n.t(), cf_n)
        print(f'dist.size={dist.size()}, dist.device={dist.device}')
        index = torch.topk(dist, expand_num, 1)[1]
        print(index.device)
        del dist
        del cf_n
        gc.collect()

        max_color_index = np.max(c_mask)
        for color_index in range(1, max_color_index + 1):
            c_pos_list = np.where(c_mask == color_index)
            c_index_list = c_pos_list[0] * wc + c_pos_list[1]
            for c_index in c_index_list:
                sim_c_index_list = index[c_index]
                for sim_c_index in sim_c_index_list:
                    x, y = sim_c_index // wc, sim_c_index % wc
                    if c_mask[x][y] == 0:
                        c_mask[x][y] = color_index
        c_mask_img = image_map1(c_mask)
        c_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 'c_mask_resized_expand.png'))
        mask = self.cal_mask(c_mask, s_mask, cf_size, sf_size, add_mask_type, False)
        return mask

    def cal_c_and_s_mask(self, c_pos_dict, s_pos_dict):
        image_size = self.cfg.TEST.GUI.IMAGE_SIZE
        c_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        s_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        color_index = 1
        print(f'Get c_pos_dict and s_pos_dict!')
        for key in c_pos_dict.keys():
            c_pos_set = c_pos_dict[key]
            s_pos_set = s_pos_dict[key]
            if c_pos_set.__len__() > 0 and s_pos_set.__len__() > 0:
                for (x, y) in c_pos_set:
                    c_mask[x][y] = color_index
                for (x, y) in s_pos_set:
                    s_mask[x][y] = color_index
                color_index += 1
        return c_mask, s_mask

    def transform_with_pos_dict(self, cf, sf, c_pos_dict, s_pos_dict, add_mask_type, expand, expand_num):
        """
        :param cf: [n, c, hc, wc]
        :param sf: [n, c, hs, ws]
        :param c_pos_dict
        :param s_pos_dict
        :param add_mask_type
        :param expand
        :param expand_num
        :return: csf [n, c, hc, wc]
        """
        ori_cf = cf.clone()
        ori_sf = sf.clone()
        ori_cf_size = ori_cf.size()
        ori_sf_size = ori_sf.size()

        cf, sf = self.down_sampling_feature(cf, sf)
        cf_size = cf.size()
        sf_size = sf.size()
        print(f'ori_cf_size={ori_cf_size}, cf_size={cf_size}, ori_sf_size={ori_sf_size}, sf_size={sf_size}')
        if not self.can_use_mask():
            return cf
        print(f'[Mast]: cal mask...')
        if not expand:
            c_mask, s_mask = self.cal_c_and_s_mask(c_pos_dict, s_pos_dict)
            c_mask_img = image_map1(c_mask)
            s_mask_img = image_map1(s_mask)
            c_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 'c_mask_origin.png'))
            s_mask_img.save(os.path.join(self.cfg.TEST.GUI.TEMP_DIR, 's_mask_origin.png'))
            mask = self.cal_mask(c_mask, s_mask, cf_size, sf_size, add_mask_type, True)
        else:
            mask = self.cal_mask_with_expand(c_pos_dict, s_pos_dict, cf, sf, add_mask_type, expand_num)
        print(f'[Mast]: mask.size={mask.size()}, mask.sum={torch.sum(mask)}, mask.type={mask.dtype}')
        if torch.sum(mask) == 0:
            del mask
            gc.collect()
            mask = None
            print(f'[Mast]: mask=None')
        patch_size = self.cfg.MAST_CORE.PATCH_SIZE
        cf_split = batch_split(cf, patch_size=(patch_size, patch_size))
        sf_split = batch_split(sf, patch_size=(patch_size, patch_size))
        ori_cf_split = batch_split(ori_cf, patch_size=(patch_size, patch_size))
        csf = self.cal_csf(ori_cf_split, cf_split, sf_split, mask)
        csf = batch_concatenate(csf, origin_size=(ori_cf_size[2], ori_cf_size[3]),
                                patch_size=(patch_size, patch_size))
        return csf


if __name__ == '__main__':
    pass
