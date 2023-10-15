import os
import glob
import numpy as np
from tqdm import tqdm
from utils import read_list, read_nifti, config
import SimpleITK as sitk
import math
import torch.nn.functional as F
import torch
import cv2

config = config.Config("mnms_2d")
base_dir = config.base_dir


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def getRangeImageDepth(label):
    d = np.any(label, axis=(1,2))
    if len(np.where(d)[0]) >0:
        d_s,d_e = np.where(d)[0][[0,-1]]
    else:
        d_s = d_e = 0

    return d_s, d_e



def split_4d_nifti(filename, save_path, npy_save_path, filename_gt, save_path_gt, npy_save_path_gt):
    img_itk = sitk.ReadImage(filename)
    lbl_itk = sitk.ReadImage(filename_gt)


    img_npy = sitk.GetArrayFromImage(img_itk)
    lbl_npy = sitk.GetArrayFromImage(lbl_itk)



    for i, t in enumerate(range(img_npy.shape[0])):
        image_arr = img_npy[t]
        label_arr = lbl_npy[t]

        if label_arr.max() > 0:


            image_arr = image_arr.astype(np.float32)
            label_arr = label_arr.astype(np.int8)
            d, h, w = image_arr.shape

            d_s, d_e = getRangeImageDepth(label_arr)

            image_arr = image_arr[d_s:d_e+1, :, :]
            label_arr = label_arr[d_s:d_e+1, :, :]


            d, _, _ = image_arr.shape

            img_itk_new = sitk.GetImageFromArray(image_arr)
            sitk.WriteImage(img_itk_new, save_path[:-7]+f'_{t}.nii.gz')
            lbl_itk_new = sitk.GetImageFromArray(label_arr)
            sitk.WriteImage(lbl_itk_new, save_path_gt[:-7]+f'_{t}.nii.gz')


            for d in range(image_arr.shape[0]):

                cur_img_slice = image_arr[d]
                cur_lbl_slice = label_arr[d]

                if image_arr[d].sum() == 0 or np.any(np.isnan(image_arr[d])):
                    print(npy_save_path+f'_{t}_{d}_image.png')
                    continue


                w_s = w // 2 - w // 3
                w_e = w // 2 + w // 3
                h_s = h // 2 - h // 3
                h_e = h // 2 + h // 3

                cur_img_slice = cur_img_slice[h_s:h_e, w_s:w_e]
                cur_lbl_slice = cur_lbl_slice[h_s:h_e, w_s:w_e]

                hn, wn = cur_img_slice.shape

                cur_img_slice = torch.FloatTensor(cur_img_slice).unsqueeze(0).unsqueeze(0)
                cur_lbl_slice = torch.FloatTensor(cur_lbl_slice).unsqueeze(0).unsqueeze(0)

                # resize_shape = (192, 192)
                if hn < wn:
                    resize_shape = (288, int(288 / hn * wn))
                else:
                    resize_shape = (int(288 / wn * hn), 288)

                cur_lbl_slice = F.interpolate(cur_lbl_slice, size=resize_shape, mode="nearest")
                cur_img_slice = F.interpolate(cur_img_slice, size=resize_shape, mode="bilinear", align_corners=False)

                cur_lbl_slice = cur_lbl_slice.squeeze().numpy()
                cur_img_slice = cur_img_slice.squeeze().numpy()

                cv2.imwrite(npy_save_path.replace('npy', 'png')+f'_{t}_{d}_image.png',cur_img_slice/cur_img_slice.max()*255)
                cv2.imwrite(npy_save_path.replace('npy', 'png')+f'_{t}_{d}_label.png',cur_lbl_slice/3*255)

                np.save(
                    npy_save_path+f'_{t}_{d}_image.npy',
                    cur_img_slice
                )
                np.save(
                    npy_save_path+f'_{t}_{d}_label.npy',
                    cur_lbl_slice
                )







def process_png():
    vendor_A = []
    vendor_B = []
    vendor_C = []
    vendor_D = []
    for tag in ['A', 'B', 'C', 'D']:
        # img_ids = []
        print('vendor', tag)
        for path in glob.glob(os.path.join(base_dir, f'vendor{tag}', '*')):
            # print(path)

            img_id = path.split('/')[-1].split('.')[0] + '_sa'
            # print(img_id)
            # if img_id == "K4T7Y0_sa":
            #     continue
            # if tag == "A":
            locals()['vendor_{}'.format(tag)].append(img_id)
            # elif tag == "C":
            #     vendor_C.append(img_id)
            # else:
            #     vendor_D.append(img_id)
        if not os.path.exists(os.path.join(config.save_dir, 'split_txts')):
            os.makedirs(os.path.join(config.save_dir, 'split_txts'))

        write_txt(
            locals()['vendor_{}'.format(tag)],
            os.path.join(config.save_dir, f'split_txts/vendor{tag}.txt')
        )


        # print(locals()['vendor_{}'.format(tag)])
        for img_id in tqdm(locals()['vendor_{}'.format(tag)]):
            label_id= img_id + '_gt'
            path = os.path.join(base_dir, f'vendor{tag}')
            if not os.path.exists(os.path.join(config.save_dir, 'processed')):
                os.makedirs(os.path.join(config.save_dir, 'processed'))

            if not os.path.exists(os.path.join(config.save_dir, 'png')):
                os.makedirs(os.path.join(config.save_dir, 'png'))
            if not os.path.exists(os.path.join(config.save_dir, 'npy')):
                os.makedirs(os.path.join(config.save_dir, 'npy'))


            image_path = os.path.join(path, img_id[:-3],  f'{img_id}.nii.gz')
            label_path =os.path.join(path, img_id[:-3], f'{label_id}.nii.gz')

            split_4d_nifti(image_path, os.path.join(config.save_dir, 'processed', f'{img_id}.nii.gz'),
                                   os.path.join(config.save_dir, 'npy', f'{img_id}'),
                                   label_path, os.path.join(config.save_dir, 'processed', f'{label_id}.nii.gz'),
                                   os.path.join(config.save_dir, 'npy', f'{label_id}'))




def split_labeled_unlabeled(ids_list, labeled_ratio):
    ids_list = np.random.permutation(ids_list)

    split_idx = math.ceil(len(ids_list) * labeled_ratio)
    print(split_idx)
    labeled_ids = sorted(ids_list[:split_idx])
    unlabeled_ids = sorted(ids_list[split_idx:])

    return labeled_ids, unlabeled_ids

def split_train_val(ids_list, val_ratio):
    ids_list = np.random.permutation(ids_list)

    split_idx = int(len(ids_list) * val_ratio)
    val_ids = sorted(ids_list[:split_idx])
    train_ids = sorted(ids_list[split_idx:])
    return train_ids, val_ids

def convert_to_true_ids(ids_list):
    true_ids = []
    for id in ids_list:
        true_path = glob.glob(os.path.join(config.save_dir, 'png', f'{id}*'))
        true_ids += [path.split('/')[-1].split('.')[0].replace("_image", "") for path in true_path if "label" not in path]
    return true_ids


def process_split(labeled_ratio=0.05):

    ids_list_A = read_list("vendorA", task="mnms")
    ids_list_B = read_list("vendorB", task="mnms")
    ids_list_C = read_list("vendorC", task="mnms")
    ids_list_D = read_list("vendorD", task="mnms")

    ids_list_A_l, ids_list_A_u = split_labeled_unlabeled(ids_list_A, labeled_ratio)
    ids_list_B_l, ids_list_B_u = split_labeled_unlabeled(ids_list_B, labeled_ratio)
    ids_list_C_l, ids_list_C_u = split_labeled_unlabeled(ids_list_C, labeled_ratio)
    ids_list_D_l, ids_list_D_u = split_labeled_unlabeled(ids_list_D, labeled_ratio)

    train_toA_l = ids_list_B_l + ids_list_C_l + ids_list_D_l
    train_toB_l = ids_list_A_l + ids_list_C_l + ids_list_D_l
    train_toC_l = ids_list_A_l + ids_list_B_l + ids_list_D_l
    train_toD_l = ids_list_A_l + ids_list_B_l + ids_list_C_l

    train_toA_u = ids_list_B_u + ids_list_C_u + ids_list_D_u
    train_toB_u = ids_list_A_u + ids_list_C_u + ids_list_D_u
    train_toC_u = ids_list_A_u + ids_list_B_u + ids_list_D_u
    train_toD_u = ids_list_A_u + ids_list_B_u + ids_list_C_u

    eval_toA = train_toA_l
    eval_toB = train_toB_l
    eval_toC = train_toC_l
    eval_toD = train_toD_l

    test_toA = ids_list_A
    test_toB = ids_list_B
    test_toC = ids_list_C
    test_toD = ids_list_D


    # ======== Vendor A =============
    write_txt(convert_to_true_ids(train_toA_l), os.path.join(config.save_dir, f'split_txts/train_toA_labeled_{labeled_ratio}.txt'))
    write_txt(convert_to_true_ids(train_toA_u), os.path.join(config.save_dir, f'split_txts/train_toA_unlabeled_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(eval_toA),   os.path.join(config.save_dir,  f'split_txts/eval_toA_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(test_toA),   os.path.join(config.save_dir,  f'split_txts/test_toA_{labeled_ratio}.txt'))

    # ======== Vendor B =============
    write_txt(convert_to_true_ids(train_toB_l), os.path.join(config.save_dir, f'split_txts/train_toB_labeled_{labeled_ratio}.txt'))
    write_txt(convert_to_true_ids(train_toB_u), os.path.join(config.save_dir, f'split_txts/train_toB_unlabeled_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(eval_toB),   os.path.join(config.save_dir,  f'split_txts/eval_toB_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(test_toB),   os.path.join(config.save_dir,  f'split_txts/test_toB_{labeled_ratio}.txt'))

    # ======== Vendor C =============
    write_txt(convert_to_true_ids(train_toC_l), os.path.join(config.save_dir, f'split_txts/train_toC_labeled_{labeled_ratio}.txt'))
    write_txt(convert_to_true_ids(train_toC_u), os.path.join(config.save_dir, f'split_txts/train_toC_unlabeled_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(eval_toC),   os.path.join(config.save_dir,  f'split_txts/eval_toC_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(test_toC),   os.path.join(config.save_dir,  f'split_txts/test_toC_{labeled_ratio}.txt'))

    # ======== Vendor D =============
    write_txt(convert_to_true_ids(train_toD_l), os.path.join(config.save_dir, f'split_txts/train_toD_labeled_{labeled_ratio}.txt'))
    write_txt(convert_to_true_ids(train_toD_u), os.path.join(config.save_dir, f'split_txts/train_toD_unlabeled_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(eval_toD),   os.path.join(config.save_dir,  f'split_txts/eval_toD_{labeled_ratio}.txt'))
    write_txt( convert_to_true_ids(test_toD),   os.path.join(config.save_dir,  f'split_txts/test_toD_{labeled_ratio}.txt'))




if __name__ == '__main__':
    process_png()
    process_split(labeled_ratio=0.02)
    process_split(labeled_ratio=0.05)
