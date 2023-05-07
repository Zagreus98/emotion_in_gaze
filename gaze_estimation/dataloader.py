from typing import Tuple

import yacs.config
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import create_dataset


def create_dataloader(
        config: yacs.config.CfgNode,
) -> Tuple[DataLoader, DataLoader]:

    train_dataset, val_dataset, sampler_weights = create_dataset(config)
    data_sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
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
