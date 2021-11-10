import torchvision.transforms as T
from torch import nn


class MNISTTransform(nn.Module):
    def __init__(self, input_size: int, train_mode: bool):
        super().__init__()
        if train_mode:
            self.transforms = nn.Sequential(
                T.Resize((input_size, input_size)),
                T.RandomAffine(10, translate=(0.02, 0.02), scale=(0.9, 1.1)),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )
        else:
            self.transforms = nn.Sequential(
                T.Resize((input_size, input_size)),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            )

    def forward(self, x):
        return self.transforms(x)
