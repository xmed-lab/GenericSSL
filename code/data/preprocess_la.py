import os
import numpy as np
from tqdm import tqdm
import h5py
from utils import read_list, read_nifti, config

config = config.Config("la")
base_dir = config.base_dir



def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    for tag in ['train', 'test']:
        img_list = []
        with open(base_dir+f'/{tag}.list', 'r') as f:
            image_list = f.readlines()
        for img_id in tqdm(image_list):
            img_id = img_id.replace('\n', '')
            print(img_id)
            img_list.append(img_id)
            path = os.path.join(base_dir, "2018LA_Seg_Training Set", img_id, "mri_norm2.h5")

            h5f = h5py.File(path)
            image = h5f['image'][:]
            label = h5f['label'][:]

            image = image.astype(np.float32)

            if not os.path.exists(os.path.join(config.save_dir, 'npy')):
                os.makedirs(os.path.join(config.save_dir, 'npy'))

            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id}_image.npy'),
                image
            )
            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id}_label.npy'),
                label
            )

            if not os.path.exists(os.path.join(config.save_dir, 'split_txts')):
                os.makedirs(os.path.join(config.save_dir, 'split_txts'))
            write_txt(
                img_list,
                os.path.join(config.save_dir, f'split_txts/{tag}.txt')
            )






def process_split_semi(split='train', labeled_ratio=0.05):
    ids_list = read_list(split, task="la")
    split_idx = int(len(ids_list) * labeled_ratio)
    labeled_ids = ids_list[:split_idx]
    unlabeled_ids = ids_list[split_idx:]
    
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, f'split_txts/labeled_{labeled_ratio}.txt')
    )
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, f'split_txts/eval_{labeled_ratio}.txt')
    )
    write_txt(
        unlabeled_ids,
        os.path.join(config.save_dir, f'split_txts/unlabeled_{labeled_ratio}.txt')
    )


if __name__ == '__main__':
    process_npy()
    process_split_semi(labeled_ratio=0.05)
    process_split_semi(labeled_ratio=0.1)
