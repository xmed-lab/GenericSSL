import random
import torch
import copy
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
import torch.nn.functional as F

from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform, BrightnessTransform
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords

import numpy as np


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                # print(item.max())
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            # elif key == 'image_raw':
            #     ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            elif key == 'label':
                # item[item>config.num_cls-1]=0
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError(key)
        # print(ret_dict['image'].shape)

        return ret_dict


def augment_rotation(data, seg, patch_size, patch_center_dist_from_border=30,
                     angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                     border_mode_data='constant', border_cval_data=0, order_data=3,
                     border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=False):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        a_x = np.random.uniform(angle_x[0], angle_x[1])
        if dim == 3:
            a_y = 0
            a_z = 0
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            coords = rotate_coords_2d(coords, a_x)

        # now find a nice center location
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 2] - patch_center_dist_from_border[d])
            else:
                ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr
        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)
    return data_result, seg_result


def augment_scale(data, seg, patch_size, patch_center_dist_from_border=30,
                  scale=(0.6, 1.0), border_mode_data='constant', border_cval_data=0, order_data=3,
                  border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=False,
                  independent_scale_for_each_axis=False, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
            sc = []
            for _ in range(dim):
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc.append(np.random.uniform(scale[0], 1))
                else:
                    sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
        else:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
        coords = scale_coords(coords, sc)

        # print("---------------------------------111")

        # now find a nice center location
        for d in range(dim):
            ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr


        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        # print("---------------------------------222")
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)
        # print("---------------------------------333")
        # print(data_result.shape, seg_result.shape)
    return data_result, seg_result


class RandomSelect(Compose):
    def __init__(self, transforms, sample_num=1):
        super(RandomSelect, self).__init__(transforms)
        self.transforms = transforms
        self.sample_num = sample_num

    def update_list(self, list):
        self.list = list

    def __call__(self, data_dict):

        tr_transforms = random.sample(self.transforms, k=self.sample_num)
        list = copy.deepcopy(self.list)
        if tr_transforms is not None:
            for i in range(len(tr_transforms)):
                list.insert(3, tr_transforms[i])
        for t in list:
            data_dict = t(**data_dict)
        del tr_transforms
        del list

        for key in data_dict.keys():
            if key == "image":
                data_dict[key] = data_dict[key].squeeze(0)
            elif key == "label":
                data_dict[key] = data_dict[key].squeeze(0).squeeze(0)

        return data_dict


class ScaleTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0, scale=(0.6, 1.0)):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.scale = scale


    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                data, label = augment_scale(data, label, patch_size=data.shape[2:4], scale=self.scale)
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict

class RotationTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                data, label = augment_rotation(data, label, patch_size=data.shape[2:4])
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ret_dict = {}
        resize_shape=(self.output_size[0],
                      self.output_size[1])
        for key in sample.keys():
            item = sample[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            if key == 'image':
                item = F.interpolate(item, size=resize_shape,mode='bilinear', align_corners=False)
            else:
                item = F.interpolate(item, size=resize_shape, mode="nearest")
            item = item.squeeze().numpy()
            ret_dict[key] = item

        return ret_dict


class RandomCrop(AbstractTransform):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, **data_dict):
        image, label = data_dict['image'], data_dict['label']
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 2, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 2, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        label = label[np.newaxis, np.newaxis, ...]
        image = image[np.newaxis, np.newaxis, ...]
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'label': label}


def get_StrongAug(patch_size, sample_num, p_per_sample=0.3):
    tr_transforms = []
    tr_transforms_select = []
    tr_transforms.append(RandomCrop(patch_size))
    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))

    # ========== Spatial-level Transforms =================
    tr_transforms_select.append(RotationTransform(p_per_sample=p_per_sample))
    tr_transforms_select.append(ScaleTransform(scale=(0.5, 1.0) , p_per_sample=p_per_sample))

    # ========== Pixel-level Transforms =================
    tr_transforms_select.append(GaussianBlurTransform((0.7, 1.3), p_per_sample=p_per_sample))
    tr_transforms_select.append(BrightnessMultiplicativeTransform(p_per_sample=p_per_sample))
    tr_transforms_select.append(ContrastAugmentationTransform(contrast_range=(0.5, 1.5), p_per_sample=p_per_sample))
    tr_transforms_select.append(GammaTransform(invert_image=False, per_channel=True, retain_stats=True, p_per_sample=p_per_sample))  # inverted gamma

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'label', True))
    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))
    trivialAug = RandomSelect(tr_transforms_select, sample_num)
    trivialAug.update_list(tr_transforms)
    return trivialAug



