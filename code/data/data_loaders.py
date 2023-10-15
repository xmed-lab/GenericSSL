import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import read_list, read_data, read_list_2d, read_data_2d


class DatasetAllTasks(Dataset):
    def __init__(self, split='train', num_cls=1, task="", repeat=None, transform=None, unlabeled=False, is_val=False, is_2d=False):
        if is_2d:
            self.ids_list = read_list_2d(split, task=task)
        else:
            self.ids_list = read_list(split, task=task)
        self.repeat = repeat
        self.task = task
        if self.repeat is None:
            self.repeat = len(self.ids_list)
        print('total {} datas'.format(self.repeat))
        self.transform = transform
        self.unlabeled = unlabeled
        self.num_cls = num_cls
        self.is_val = is_val
        self.is_2d = is_2d
        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list):
                if is_2d:
                    image, label = read_data(data_id, task=task+"_2d")
                else:
                    image, label = read_data(data_id, task=task)
                self.data_list[data_id] = (image, label)

    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):
        if self.is_val:
            image, label = self.data_list[data_id]
        else:
            if self.is_2d:
                image, label = read_data(data_id, task=self.task+"_2d")
            else:
                image, label = read_data(data_id, task=self.task)
        return data_id, image, label


    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)

        if self.unlabeled: # <-- for safety
            label[:] = 0
        if "synapse" in self.task:
            image = image.clip(min=-75, max=275)

        elif "mnms" in self.task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / (image.max() - image.min())

        image = image.astype(np.float32)
        label = label.astype(np.int8)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

