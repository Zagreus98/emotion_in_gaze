from typing import Tuple, Union

import yacs.config
from torch.utils.data import DataLoader

from .datasets import create_dataset


def create_dataloader(
        config: yacs.config.CfgNode,
        is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        # TODO: create weights for each dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.train_dataloader.num_workers,

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
