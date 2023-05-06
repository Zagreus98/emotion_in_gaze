from torch.utils.data import Dataset
import pathlib
from PIL import Image
import pandas as pd
import numpy as np
import torch


class RafDataset(Dataset):
    def __init__(self, raf_path, transform, mode):
        self.transform = transform
        self.raf_path = raf_path
        self.mode = mode
        self.dataset = self.get_rafdb()
        self.random_horizontal_flip = False
        self.raf2affect = {
            0: 6,
            1: 2,
            2: 1,
            3: 3,
            4: 5,
            5: 0,
            6: 4,
        }

    def get_image(self, img_name):
        img_path = pathlib.Path.joinpath(self.raf_path, 'aligned', img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_rafdb(self):
        base_dataset = pd.read_csv(
            pathlib.Path.joinpath(self.raf_path, 'list_patition_label.txt'),
            sep=' ',
            header=None,
            names=['img', 'label'],
        )
        # change the names to actual names of the images and make labels start from 0
        add_align = lambda x: str(x).split('.')[0] + '_aligned.jpg'
        base_dataset['img'] = base_dataset['img'].apply(add_align)
        base_dataset['label'] = base_dataset['label'] - 1
        if self.mode == 'train':
            dataset = base_dataset[base_dataset['img'].str.startswith('train')]
        else:  # val
            dataset = base_dataset[base_dataset['img'].str.startswith('test')]

        return dataset

    def __getitem__(self, idx):

        img_name = self.dataset.iloc[idx, 0]
        emotion = torch.Tensor([self.raf2affect[self.dataset.iloc[idx, 1]]]).to(torch.int64)
        image = self.get_image(img_name)
        if self.random_horizontal_flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
        gaze = torch.Tensor([0, 0, 0])  # (ignore flag , [pitch, yaw])

        return image, gaze, emotion

    def __len__(self):
        return len(self.dataset)
