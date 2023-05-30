import os

import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import cv2

class FerDataset(Dataset):
    def __init__(self, fer_path, transform, mode):
        self.transform = transform
        self.fer_path = fer_path
        self.mode = mode
        self.class_to_idx = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6,
        }
        self.dataset = self.get_fer()
        self.random_horizontal_flip = False

    def get_fer(self):
        dataset = []
        if self.mode == 'train':
            data_root_path = Path.joinpath(self.fer_path, 'train')
        else:
            data_root_path = Path.joinpath(self.fer_path, 'test')

        emotion_dirs = data_root_path.glob('**/*')
        for emotion_dir in emotion_dirs:
            if emotion_dir.is_dir():
                images = emotion_dir.glob('*.png')
                for img_path in images:
                    sep = os.sep
                    emotion_name = str(img_path).split(sep)[-2]
                    if emotion_name == 'contempt':
                        continue
                    emotion_idx = self.class_to_idx[emotion_name]
                    target = {
                        'img_path': str(img_path),
                        'emotion': emotion_idx,
                        'gaze': [0.0, 0.0, 0.0]
                    }
                    dataset.append(target)
        return dataset

    def __getitem__(self, idx):

        target = self.dataset[idx]
        image = cv2.imread(target['img_path'])[..., ::-1]
        if self.random_horizontal_flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
        image = self.transform(image)
        gaze = torch.tensor(target['gaze'])
        emotion = torch.Tensor([target['emotion']]).to(torch.int64)

        return image, gaze, emotion

    def __len__(self):
        return len(self.dataset)
