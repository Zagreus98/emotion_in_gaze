import pathlib
from typing import List, Union, Tuple

import json
import yacs.config
from torch.utils.data import Dataset, ConcatDataset

from ..transforms import create_xgaze_transform, create_rafdb_tranform
from .rafdb import RafDataset
from .ethxgaze import OnePersonDataset


def create_dataset(
        config: yacs.config.CfgNode,
) -> Union[List[Dataset], Dataset, Tuple[ConcatDataset, ConcatDataset, List]]:

    # Initialize dataset paths
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)
    raf_dir = pathlib.Path(config.dataset.raf_dataset_path)
    assert dataset_dir.exists()
    assert raf_dir.exists()

    train_dataset = []
    val_dataset = []
    sampler_weights = []

    # ETH-XGaze datasets reading
    split_file = dataset_dir / 'train_test_split.json'
    with open(split_file) as f:
        split = json.load(f)
    train_paths = [
        dataset_dir / 'train' / name for name in split['train']
    ]
    for path in train_paths:
        assert path.exists()

    # Define gaze transforms
    gaze_train_transform = create_xgaze_transform(config, 'train')
    gaze_val_transform = create_xgaze_transform(config, 'val')
    # Define emotion transforms
    emo_train_transfrom = create_rafdb_tranform(config, 'train')
    emo_val_transfrom = create_rafdb_tranform(config, 'val')
    # Get val indices for person datasets on which to perform evaluation
    val_indices = config.train.val_indices
    assert val_indices is not None

    # Prepare train dataset
    gaze_train_datasets = [
        OnePersonDataset(path, gaze_train_transform)
        for i, path in enumerate(train_paths)
        if i not in val_indices
    ]
    emotion_train_dataset = RafDataset(
        raf_path=raf_dir,
        transform=emo_train_transfrom,
        mode='train',
    )
    train_dataset.append(emotion_train_dataset)
    train_dataset.extend(gaze_train_datasets)
    for i, dataset in enumerate(train_dataset):
        if i == 0:
            weight = 10 / len(dataset)  # TODO: add a hyperparameter for this
        else:
            weight = 1 / len(dataset)
        sampler_weights.extend([weight] * len(dataset))

    train_dataset = ConcatDataset(train_dataset)

    # Prepare val dataset
    gaze_val_dataset = [
        OnePersonDataset(path, gaze_val_transform)
        for i, path in enumerate(train_paths)
        if i in val_indices
    ]
    emotion_val_dataset = RafDataset(
        raf_path=raf_dir,
        transform=emo_val_transfrom,
        mode='val',
    )
    val_dataset.append(emotion_val_dataset)
    val_dataset.extend(gaze_val_dataset)
    val_dataset = ConcatDataset(val_dataset)

    # Initialize horizontal flip for all the datasets
    if config.dataset.transform.train.horizontal_flip:
        for dataset in train_dataset.datasets:
            dataset.random_horizontal_flip = True

    return train_dataset, val_dataset, sampler_weights
