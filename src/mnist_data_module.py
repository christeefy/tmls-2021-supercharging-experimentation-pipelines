from multiprocessing import cpu_count

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .mnist_dataset import MNISTDataset


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
    ):
        self.batch_size = batch_size
        self.num_workers = cpu_count() - 1
        self.train_dataset = MNISTDataset("train")
        self.val_dataset = MNISTDataset("val")
        self.test_dataset = MNISTDataset("test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
