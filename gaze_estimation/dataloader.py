from typing import Tuple, Union

import yacs.config
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import create_dataset


def create_dataloader(
        config: yacs.config.CfgNode,
        is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        train_dataset, val_dataset, sampler_weights = create_dataset(config, is_train)
        data_sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        # TODO: create weights for each dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.train_dataloader.num_workers,
            sampler=data_sampler,

        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.val_dataloader.num_workers,
        )
        return train_loader, val_loader
    else:
        test_dataset = create_dataset(config, is_train)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test.batch_size,
            num_workers=config.test.dataloader.num_workers,
            shuffle=False,
            pin_memory=config.test.dataloader.pin_memory,
            drop_last=False,
        )
        return test_loader
