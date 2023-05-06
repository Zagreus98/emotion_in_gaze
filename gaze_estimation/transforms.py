from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
import yacs.config

from .types import GazeEstimationMethod


def create_transform(config: yacs.config.CfgNode) -> Any:
    if config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        return _create_mpiifacegaze_transform(config)
    else:
        raise ValueError


def create_xgaze_transform(config: yacs.config.CfgNode, stage: str) -> Any:
    assert stage in ['train', 'val', 'test']
    mean = np.array(config.dataset.mean)
    std = np.array(config.dataset.std)
    transform_config = getattr(config.dataset.transform, stage)

    transforms = [
        T.ToPILImage(),
    ]
    if ('resize' in transform_config
            and transform_config.resize != config.dataset.image_size):
        transforms.append(T.Resize(transform_config.resize))
    if 'color_jitter' in transform_config:
        transforms.append(
            T.ColorJitter(brightness=transform_config.color_jitter.brightness,
                          contrast=transform_config.color_jitter.contrast,
                          saturation=transform_config.color_jitter.saturation,
                          hue=transform_config.color_jitter.hue))

    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean, std, inplace=False),
    ])

    return T.Compose(transforms)


def create_rafdb_tranform(config: yacs.config.CfgNode, stage: str) -> Any:
    assert stage in ['train', 'val', 'test']
    mean = np.array(config.dataset.mean)
    std = np.array(config.dataset.std)
    transform_config = getattr(config.dataset.transform, stage)

    if stage == 'train':
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize(transform_config.resize),
            T.ToTensor(),
            T.Normalize(mean, std, inplace=False),
            T.RandomErasing(scale=(0.02, 0.25))
        ])
    else:
        transforms = T.Compose([
            T.Resize(transform_config.resize),
            T.ToTensor(),
            T.Normalize(mean, std, inplace=False),
        ])

    return transforms


def _create_mpiifacegaze_transform(config: yacs.config.CfgNode) -> Any:
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    identity = torchvision.transforms.Lambda(lambda x: x)
    size = config.transform.mpiifacegaze_face_size
    if size != 448:
        resize = torchvision.transforms.Lambda(
            lambda x: cv2.resize(x, (size, size)))
    else:
        resize = identity
    if config.transform.mpiifacegaze_gray:
        to_gray = torchvision.transforms.Lambda(lambda x: cv2.cvtColor(
            cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.
            COLOR_GRAY2BGR))
    else:
        to_gray = identity

    transform = torchvision.transforms.Compose([
        resize,
        to_gray,
        torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)),
        scale,
        torch.from_numpy,
        torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229]),
    ])
    return transform
