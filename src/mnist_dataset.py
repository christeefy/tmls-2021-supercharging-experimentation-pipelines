from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        labels_path: Optional[str] = None,
    ):
        self.images = torch.concat([torch.load(images_path)] * 3, dim=1)
        self.labels = torch.load(labels_path) if labels_path is not None else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, index
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.labels is not None:
            return self.images[index], self.labels[index]
        return self.images[index]
