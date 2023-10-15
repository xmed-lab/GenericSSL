import os
import glob
import numpy as np
from tqdm import tqdm
from utils import read_list, read_nifti, config
import SimpleITK as sitk
import itk
from scipy.ndimage import zoom



config = config.Config("mmwhs")
base_dir = config.base_dir


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def convert_labels(label):
    label[label==205] = 1
    label[label==420] = 2
    label[label==500] = 3
    label[label==820] = 4
    label[label>4] = 0
    return label


def read_reorient2RAI(path):
    itk_img = itk.imread(path)

    filter = itk.OrientImageFilter.New(itk_img)
    filter.UseImageDirectionOn()
    filter.SetInput(itk_img)
    m = itk.Matrix[itk.D, 3, 3]()
    m.SetIdentity()
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    itk_img = filter.GetOutput()

    itk_arr = itk.GetArrayViewFromImage(itk_img)
    return itk_arr




def getRangeImageDepth(label):
    d = np.any(label, axis=(1,2))
    h = np.any(label, axis=(0,2))
    w = np.any(label, axis=(0,1))

    if len(np.where(d)[0]) >0:
        d_s, d_e = np.where(d)[0][[0,-1]]
    else:
        d_s = d_e = 0

    if len(np.where(h)[0]) >0:
        h_s,h_e = np.where(h)[0][[0,-1]]
    else:
        h_s = h_e = 0

    if len(np.where(w)[0]) >0:
        w_s,w_e = np.where(w)[0][[0,-1]]
    else:
        w_s = w_e = 0
    return d_s, d_e, h_s, h_e, w_s, w_e



def process_npy():
    for tag in ['MR', 'CT']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, tag, f'imagesTr', '*.nii.gz'))):
            img_id = path.split('/')[-1].split('.')[0]
            print(img_id)
            img_ids.append(img_id)

            label_id= img_id[:-5] + 'label'

            image_path = os.path.join(base_dir,tag, f'imagesTr', f'{img_id}.nii.gz')
            label_path =os.path.join(base_dir,tag, f'labelsTr', f'{label_id}.nii.gz')

            if not os.path.exists(os.path.join(config.save_dir, 'processed')):
                os.makedirs(os.path.join(config.save_dir, 'processed'))

            image_arr = read_reorient2RAI(image_path)
            label_arr = read_reorient2RAI(label_path)

            image_arr = image_arr.astype(np.float32)
            label_arr = convert_labels(label_arr)

            if img_id == "mr_train_1002_image":
                label_arr[0:4, :, :] = 0
                label_arr[:, -10:-1, :] = 0
                label_arr[:, :, 0:4] = 0



            d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(label_arr)
            d, h, w = image_arr.shape

            d_s = (d_s - 4).clip(min=0, max=d)
            d_e = (d_e + 4).clip(min=0, max=d)
            h_s = (h_s - 4).clip(min=0, max=h)
            h_e = (h_e + 4).clip(min=0, max=h)
            w_s = (w_s - 4).clip(min=0, max=w)
            w_e = (w_e + 4).clip(min=0, max=w)

            image_arr = image_arr[d_s:d_e, h_s:h_e, w_s: w_e]
            label_arr = label_arr[d_s:d_e, h_s:h_e, w_s: w_e]

            upper_bound_intensity_level = np.percentile(image_arr, 98)

            image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
            image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)

            dn, hn, wn = image_arr.shape


            image_arr = zoom(image_arr, [144/dn, 144/hn, 144/wn], order=0)
            label_arr = zoom(label_arr, [144/dn, 144/hn, 144/wn], order=0)


            image = sitk.GetImageFromArray(image_arr)
            label = sitk.GetImageFromArray(label_arr)

            sitk.WriteImage(image, os.path.join(config.save_dir, 'processed', f'{img_id}.nii.gz'))
            sitk.WriteImage(label, os.path.join(config.save_dir, 'processed', f'{label_id}.nii.gz'))


            if not os.path.exists(os.path.join(config.save_dir, 'npy')):
                os.makedirs(os.path.join(config.save_dir, 'npy'))

            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id}.npy'),
                image_arr
            )
            np.save(
                os.path.join(config.save_dir, 'npy', f'{label_id}.npy'),
                label_arr
            )



def process_split_fully(train_val_ratio=0.8):
    ct_img_ids = []
    mr_img_ids = []
    for path in tqdm(glob.glob(os.path.join(config.save_dir, 'npy', '*_image.npy'))):
        img_id = path.split('/')[-1].split('.')[0][:-6]
        if 'mr' in img_id:
            mr_img_ids.append(img_id)
        else:
            ct_img_ids.append(img_id)

    print(len(mr_img_ids), mr_img_ids)
    print(len(ct_img_ids), ct_img_ids)

    test_mr_ids = ["mr_train_1007",
                   "mr_train_1009",
                   "mr_train_1018",
                   "mr_train_1019"]
    test_ct_ids = ["ct_train_1003",
                   "ct_train_1008",
                   "ct_train_1014",
                   "ct_train_1019"]

    mr_train_val_ids = np.setdiff1d(mr_img_ids, test_mr_ids)
    ct_train_val_ids = np.setdiff1d(ct_img_ids, test_ct_ids)

    print(mr_train_val_ids)
    print(ct_train_val_ids)


    mr_train_val_ids = np.random.permutation(mr_train_val_ids)
    ct_train_val_ids = np.random.permutation(ct_train_val_ids)

    split_idx = round(len(mr_train_val_ids) * train_val_ratio)
    mr_train_ids = sorted(mr_train_val_ids[:split_idx])
    ct_train_ids = sorted(ct_train_val_ids[:split_idx])
    mr_eval_ids = sorted(mr_train_val_ids[split_idx:])
    ct_eval_ids = sorted(ct_train_val_ids[split_idx:])

    train_mr2ct_labeled = mr_train_val_ids
    train_mr2ct_unlabeled = ct_train_ids
    train_ct2mr_labeled = ct_train_val_ids
    train_ct2mr_unlabeled = mr_train_ids
    eval_mr2ct = ct_eval_ids
    eval_ct2mr = mr_eval_ids

    print("======= MR to CT ===========")
    print(len(train_mr2ct_labeled), train_mr2ct_labeled)
    print(len(train_mr2ct_unlabeled), train_mr2ct_unlabeled)
    print(eval_mr2ct)


    print("======= CT to MR ===========")
    print(len(train_ct2mr_labeled), train_ct2mr_labeled)
    print(len(train_ct2mr_unlabeled), train_ct2mr_unlabeled)
    print(eval_ct2mr)


    if not os.path.exists(os.path.join(config.save_dir, 'split_txts')):
        os.makedirs(os.path.join(config.save_dir, 'split_txts'))

    write_txt(
        train_mr2ct_labeled,
        os.path.join(config.save_dir, 'split_txts/train_mr2ct_labeled.txt')
    )
    write_txt(
        train_mr2ct_unlabeled,
        os.path.join(config.save_dir, 'split_txts/train_mr2ct_unlabeled.txt')
    )
    write_txt(
        eval_mr2ct,
        os.path.join(config.save_dir, 'split_txts/eval_mr2ct.txt')
    )


    write_txt(
        train_ct2mr_labeled,
        os.path.join(config.save_dir, 'split_txts/train_ct2mr_labeled.txt')
    )
    write_txt(
        train_ct2mr_unlabeled,
        os.path.join(config.save_dir, 'split_txts/train_ct2mr_unlabeled.txt')
    )
    write_txt(
        eval_ct2mr,
        os.path.join(config.save_dir, 'split_txts/eval_ct2mr.txt')
    )

    write_txt(
        test_mr_ids,
        os.path.join(config.save_dir, 'split_txts/test_mr.txt')
    )

    write_txt(
        test_ct_ids,
        os.path.join(config.save_dir, 'split_txts/test_ct.txt')
    )






if __name__ == '__main__':
    process_npy()
    process_split_fully()
