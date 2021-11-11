import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.mnist_dataset import MNISTDataset
from src.model import Model
from src.transforms import MNISTTransform

CONFIG_PATH = "predict_config.yaml"


def load_config(path):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)

    model = Model.load_from_checkpoint(
        config["checkpoint_path"],
        model_name="mobilenet",
        n_classes=10,
        transforms=MNISTTransform(224, train_mode=True),
        eval_transforms=MNISTTransform(224, train_mode=True),
    )
    inference_dataset = MNISTDataset(config["predict_images"])
    inference_dataloader = DataLoader(
        inference_dataset, batch_size=config["batch_size"]
    )

    predictions_list = Trainer().predict(model, inference_dataloader)
    predictions = torch.cat(predictions_list)
    torch.save(predictions, "predictions.pt")
