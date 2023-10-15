import os
import glob
import numpy as np
from tqdm import tqdm
from utils import read_list, read_nifti, config
import SimpleITK as sitk
import torch.nn.functional as F
import torch


config = config.Config("synapse")
base_dir = config.base_dir


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    for tag in ['Tr']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, f'images{tag}', '*.nii.gz'))):
            img_id = path.split('/')[-1].split('.')[0]
            img_ids.append(img_id)
            label_id= 'label'+img_id[3:]

            image_path = os.path.join(base_dir, f'images{tag}', f'{img_id}.nii.gz')
            label_path =os.path.join(base_dir, f'labels{tag}', f'{label_id}.nii.gz')

            image = read_nifti(image_path)
            label = read_nifti(label_path)


            resize_shape=(config.patch_size[0]+config.patch_size[0]//4,
                          config.patch_size[1]+config.patch_size[1]//4,
                          config.patch_size[2]+config.patch_size[2]//4)

            image = image.astype(np.float32)
            label = label.astype(np.int8)

            image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
            label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)

            image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
            label = F.interpolate(label, size=resize_shape,mode='nearest')
            image = image.squeeze().numpy()
            label = label.squeeze().numpy()


            if not os.path.exists(os.path.join(config.save_dir, 'processed')):
                os.makedirs(os.path.join(config.save_dir, 'processed'))


            image_nii = sitk.GetImageFromArray(image)
            label_nii = sitk.GetImageFromArray(label)

            sitk.WriteImage(image_nii, os.path.join(config.save_dir, 'processed', f'{img_id}.nii.gz'))
            sitk.WriteImage(label_nii, os.path.join(config.save_dir, 'processed', f'{label_id}.nii.gz'))

            if not os.path.exists(os.path.join(config.save_dir, 'npy')):
                os.makedirs(os.path.join(config.save_dir, 'npy'))


            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id[3:]}_image.npy'),
                image
            )
            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id[3:]}_label.npy'),
                label
            )




def process_split_fully(train_ratio=0.8):


    if not os.path.exists(os.path.join(config.save_dir, 'split_txts')):
        os.makedirs(os.path.join(config.save_dir, 'split_txts'))

    for tag in ['Tr']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, f'images{tag}', '*.nii.gz'))):
            img_id = path.split('/')[-1].split('.')[0][3:]
            img_ids.append(img_id)
        
        if tag == 'Tr':
            img_ids = np.random.permutation(img_ids)
            split_idx = int(len(img_ids) * train_ratio)
            train_val_ids = img_ids[:split_idx]
            test_ids = sorted(img_ids[split_idx:])

            split_idx = int(len(train_val_ids) * 5/6)
            train_ids = sorted(train_val_ids[:split_idx])
            eval_ids = sorted(train_val_ids[split_idx:])
            write_txt(
                train_ids,
                os.path.join(config.save_dir, 'split_txts/train.txt')
            )
            write_txt(
                eval_ids,
                os.path.join(config.save_dir, 'split_txts/eval.txt')
            )

            test_ids = sorted(test_ids)
            write_txt(
                test_ids,
                os.path.join(config.save_dir, 'split_txts/test.txt')
            )


def process_split_semi(split='train', labeled_ratio=0.2):
    ids_list = read_list(split, task="synapse")
    ids_list = np.random.permutation(ids_list)

    split_idx = int(len(ids_list) * labeled_ratio)
    labeled_ids = sorted(ids_list[:split_idx])
    unlabeled_ids = sorted(ids_list[split_idx:])
    
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, f'split_txts/labeled_{labeled_ratio}.txt')
    )
    write_txt(
        unlabeled_ids,
        os.path.join(config.save_dir, f'split_txts/unlabeled_{labeled_ratio}.txt')
    )


if __name__ == '__main__':
    process_npy()
    process_split_fully()
    process_split_semi(labeled_ratio=0.2)
