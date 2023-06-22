import pathlib
from typing import Callable
import numpy as np

import h5py
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class OnePersonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: pathlib.Path, transform: Callable):
        self.dataset_path = dataset_path
        self.transform = transform
        self.random_horizontal_flip = False
        self._length = self._get_length()

    def _get_length(self) -> int:
        with h5py.File(self.dataset_path, 'r', swmr=True) as f:
            length = len(f['face_patch'])
        return length

    def __getitem__(
            self,
            index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r', swmr=True) as f:
            image = f['face_patch'][index]
            gaze = f['face_gaze'][index]
        if self.random_horizontal_flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
            gaze *= np.array([1, -1])
        image = image[..., ::-1]  # BGR to RGB
        image = self.transform(image)
        # insert ignore flag
        gaze = np.insert(gaze, obj=0, values=1.0)
        gaze = torch.from_numpy(gaze)  # (ignore flag , [pitch, yaw])
        emotion = torch.Tensor([-1]).to(torch.int64)  # ignore class
        return image, gaze.float(), emotion

    def __len__(self) -> int:
        return self._length
