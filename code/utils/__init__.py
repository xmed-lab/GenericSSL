import os
import math
from tqdm import tqdm
import numpy as np
import random
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.config import Config
import numbers
import math
import cv2
import h5py

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.cuda(), groups=self.groups, padding="same")


def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def print_func(item):
    if type(item) == torch.Tensor:
        return [round(x,2) for x in item.data.cpu().numpy().tolist()]
    elif type(item) == np.ndarray:
        return [round(x,2) for x in item.tolist()]
    else:
        raise TypeError


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / (x_sum)
    return s

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)

    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr



def read_list(split, task):

    config = Config(task)
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'split_txts', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data(data_id, task, normalize=False):
    config = Config(task)
    im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')
    lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
    if not os.path.exists(im_path) or not os.path.exists(lb_path):
        print(im_path)
        print(lb_path)
        raise ValueError(data_id)
    image = np.load(im_path)
    label = np.load(lb_path)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / (image.max() - image.min())

        image = image.astype(np.float32)

    return image, label


def read_list_2d(split, task):

    config = Config(task+"_2d")
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'split_txts', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data_2d(data_id, task, normalize=False):
    config = Config(task+"_2d")
    if "acdc" in task:
        h5File = h5py.File(os.path.join(config.save_dir, 'h5', f'{data_id}.h5'), 'r')
        image = h5File["image"][:]
        label = h5File["label"][:]
    else:
        im_path = os.path.join(config.save_dir, 'png', f'{data_id}_image.png')
        lb_path = os.path.join(config.save_dir, 'png', f'{data_id}_label.png')
        if not os.path.exists(im_path) or not os.path.exists(lb_path):
            print(im_path)
            print(lb_path)
            raise ValueError(data_id)
        image = cv2.imread(im_path, 0)
        label = cv2.imread(lb_path, 0)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / (image.max() - image.min())

        image = image.astype(np.float32)

    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image

def test_all_case(task, net, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task=task, normalize=True)

        pred, _ = test_single_case(
            net,
            image,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):

    padding_flag = image.shape[0] < patch_size[0] or image.shape[1] < patch_size[1] or image.shape[2] < patch_size[2]
    if padding_flag:
        pw = max((patch_size[0] - image.shape[0]) // 2 + 1, 0)
        ph = max((patch_size[1] - image.shape[1]) // 2 + 1, 0)
        pd = max((patch_size[2] - image.shape[2]) // 2 + 1, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    image = image[np.newaxis]
    _, dd, ww, hh = image.shape


    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                # y1, _, _, _ = net(test_patch) # <--
                y1 = net(test_patch, pred_type="D_theta_u")
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map



def test_all_case_2d(task, net, ids_list, num_classes, patch_size, stride_xy, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, label = read_data(data_id, task=task+"_2d", normalize=True)
        label = label.astype(np.uint8)
        print(np.unique(label))
        pred, label = test_single_case_2d(
            net,
            image, label,
            stride_xy,
            patch_size,
            num_classes=num_classes
        )
        cv2.imwrite(f'{test_save_path}/{data_id}.png', pred/3*255)
        cv2.imwrite(f'{test_save_path}/{data_id}_label.png', label/3*255)




def test_single_case_2d(net, image, label, stride_xy, patch_size, num_classes):
    padding_flag = image.shape[0] <= patch_size[0] or image.shape[1] <= patch_size[1]
    # pad the sample if necessary
    if padding_flag:
        pw = max((patch_size[0] - image.shape[0]) // 2 + 1, 0)
        ph = max((patch_size[1] - image.shape[1]) // 2 + 1, 0)
        # if padding_flag:
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

    image = image[np.newaxis]

    _, hh, ww = image.shape

    sx = math.ceil((hh - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((ww - patch_size[1]) / stride_xy) + 1

    # print(sx, sy)

    score_map = np.zeros((num_classes, ) + image.shape[1:3]).astype(np.float32)
    cnt = np.zeros(image.shape[1:3]).astype(np.float32)
    for x in range(sx):
        xs = min(stride_xy*x, hh-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, ww-patch_size[1])
            test_patch = image[:,  xs:xs+patch_size[0], ys:ys+patch_size[1]]
            test_patch = torch.from_numpy(test_patch).cuda().float()
            # print(test_patch.shape)
            y1 = net(test_patch.unsqueeze(0), pred_type="D_theta_u")
            y = F.softmax(y1, dim=1) # <--
            y = y.cpu().data.numpy()
            y = y[0, ...]
            score_map[:,xs:xs+patch_size[0], ys:ys+patch_size[1]] += y
            cnt[xs:xs+patch_size[0], ys:ys+patch_size[1]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    label_map = np.argmax(score_map, axis=0)
    # print(label_map.shape)
    return label_map, label
