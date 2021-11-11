import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision import models

from .transforms import MNISTTransform


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        transforms: MNISTTransform,
        eval_transforms: MNISTTransform,
    ):
        super().__init__()
        self.metrics = Accuracy(num_classes=n_classes, average=None)
        self.transforms = (
            transforms  # Does PL set this device  # Should I put this here or in CPUs?
        )
        self.eval_transforms = eval_transforms
        self.model = self._load_model(model_name, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        loss, acc = self._forward_loss_metrics(batch, train_mode=True)
        self.log("train_loss", loss)
        self.log_dict({f"train_acc_class_{i}": v.item() for i, v in enumerate(acc)})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._forward_loss_metrics(batch, train_mode=False)
        self.log("val_loss", loss)
        self.log_dict({f"val_acc_class_{i}": v.item() for i, v in enumerate(acc)})
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._forward_loss_metrics(batch, train_mode=False)
        self.log("test_loss", loss)
        self.log_dict({f"test_acc_class_{i}": v.item() for i, v in enumerate(acc)})
        return loss

    def predict_step(self, batch, batch_idx):
        X = batch
        X = self.eval_transforms(X)
        predictions = self.model(X).softmax(dim=-1)
        return predictions

    def _forward_loss_metrics(self, batch, train_mode: bool):
        images, labels = batch
        transformed_images = (
            self.transforms(images) if train_mode else self.eval_transforms(images)
        )
        logits = self.model(transformed_images)
        loss = self.criterion(logits, labels)
        accuracy = self.metrics(logits.softmax(dim=-1), labels)
        accuracy = torch.nan_to_num(accuracy, nan=0.5)

        return loss, accuracy

    @staticmethod
    def _load_model(model_name: str, n_classes: int):
        if model_name == "resnet":
            model = models.resnet50(pretrained=True)
            n_features_in = model.fc.in_features
            model.fc = nn.Linear(n_features_in, 10)
        elif model_name == "mobilenet":
            model = models.mobilenet.mobilenet_v3_small(pretrained=True)
            n_features_in = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(n_features_in, n_classes)
        else:
            raise NotImplementedError
        return model
