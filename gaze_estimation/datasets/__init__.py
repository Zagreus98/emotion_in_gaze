import pathlib
from typing import List, Union, Tuple

import torch
import json
import yacs.config
from torch.utils.data import Dataset, ConcatDataset

from ..transforms import create_transform, create_xgaze_transform, create_rafdb_tranform
from ..types import GazeEstimationMethod
from .rafdb import RafDataset


def create_dataset(
        config: yacs.config.CfgNode,
        is_train: bool = True,
) -> Union[List[Dataset], Dataset, Tuple[ConcatDataset, ConcatDataset]]:
    # TODO: remove all MPIIFaceGaze mentions and if elses
    if config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        from .mpiifacegaze import OnePersonDataset
    elif config.mode == GazeEstimationMethod.ETHXGaze.name:
        from .ethxgaze import OnePersonDataset
    else:
        raise ValueError

    # Initialize paths
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)
    raf_dir = pathlib.Path(config.dataset.raf_dataset_path)
    assert dataset_dir.exists()
    assert raf_dir.exists()

    if config.mode == GazeEstimationMethod.ETHXGaze.name:
        train_dataset = []
        val_dataset = []

        # ETH-XGaze datatasets reading
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
        emo_val_transfrom = create_rafdb_tranform(config, 'train')
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

        return train_dataset, val_dataset

    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        transform = create_transform(config)
        assert config.train.test_id in range(-1, 15)
        assert config.test.test_id in range(15)
        ## index:02 practic iti pune cate 0 sa ai inainte de index = > p01 cand index=1
        ## daca aveam index:03   ==> p001
        person_ids = [f'p{index:02}' for index in range(15)]

        if is_train:
            # daca vren sa folosim tot dataset-ul pentru antrenare
            if config.train.test_id == -1:
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_dir, transform)
                    for person_id in person_ids
                ])
                assert len(train_dataset) == 45000
            # daca vrem sa pastram p00 pentru testare
            else:
                test_person_id = person_ids[config.train.test_id]
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_dir, transform)
                    for person_id in person_ids if person_id != test_person_id
                ])
                assert len(train_dataset) == 42000

            # cat la suta din datele de training le folosim pentru testare
            val_ratio = config.train.val_ratio
            assert val_ratio < 1
            val_num = int(len(train_dataset) * val_ratio)
            train_num = len(train_dataset) - val_num
            lengths = [train_num, val_num]
            return torch.utils.data.dataset.random_split(train_dataset, lengths)
        # daca suntem in faza de testare si cream un dataloader pentru evaluare
        else:
            test_person_id = person_ids[config.test.test_id]
            test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform)
            assert len(test_dataset) == 3000
            return test_dataset

    else:
        raise ValueError('Bad dataset')
