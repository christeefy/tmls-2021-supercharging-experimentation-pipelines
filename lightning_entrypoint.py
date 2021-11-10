from pytorch_lightning.utilities.cli import LightningCLI

from src.mnist_data_module import MNISTDataModule
from src.model import Model

if __name__ == "__main__":
    LightningCLI(Model, MNISTDataModule)
