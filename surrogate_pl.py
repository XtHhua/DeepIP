import os
import argparse
from argparse import Namespace
import sys

sys.path.append("/data/xuth/deep_ipr")

import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchvision.models import vgg16_bn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from new2024.util.data_adapter import SplitDataConverter
from new2024.util.data_adapter import defend_attack_split
from new2024.util.metric import ClassificationMetric
from new2024.config.config import PROJECT_ROOT_DIR
from new2024.source_pl import CVModel


class SurrogateModel(pl.LightningModule):
    def __init__(self, model: nn.Module, args: Namespace):
        super().__init__()
        #
        self.teacher = CVModel.load_from_checkpoint(
            os.path.join(PROJECT_ROOT_DIR, "model", "source", args.dataset_name)
        )
        self.model = model
        #
        self.args = args
        # loss
        self.train_criterion = CrossEntropyLoss()
        self.val_criterion = CrossEntropyLoss()
        # metric
        self.train_metric = ClassificationMetric(recall=False, precision=False)
        self.val_metric = ClassificationMetric(recall=False, precision=False)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        teacher_output = self.teacher(x)
        pred = torch.max(teacher_output, 1)[1].detach().squeeze()
        loss = self.train_criterion(_y, pred)
        self.train_metric.update(_y, y)
        return loss

    def training_epoch_end(self, outs):
        loss = 0.0
        for out in outs:
            loss += out["loss"].cpu().detach().item()
        loss /= len(outs)
        acc, f1 = self.train_metric.compute()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("f1", f1)

    def validation_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        val_loss = self.val_criterion(_y, y)
        self.val_metric.update(_y, y)
        return val_loss

    def validation_epoch_end(self, outs):
        val_loss = sum(outs).item() / len(outs)
        val_acc, val_f1 = self.val_metric.compute()
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        self.log("val_f1", val_f1)

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )


def main(args: Namespace):
    trainer_params = {
        "accelerator": "gpu",
        "devices": [args.gpu],
        "max_epochs": args.epochs,  # 1000
        "enable_checkpointing": True,  # True
        "logger": True,  # TensorBoardLogger
        "default_root_dir": os.path.join(
            PROJECT_ROOT_DIR, "model", args.model_type, args.dataset_name
        ),
        "progress_bar_refresh_rate": 1,  # 1
        "num_sanity_val_steps": 0,  # 2
    }
    # prepare data
    train_dataset, dev_dataset, test_dataset = SplitDataConverter().split(
        dataset_name=args.dataset_name
    )
    defend_dataset, attack_dataset = defend_attack_split(train_dataset)
    train_dataloader = DataLoader(
        defend_dataset, batch_size=args.batch_size, shuffle=True
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # prepare model
    for i in range(10):
        base_model = vgg16_bn(weights=None)
        model = CVModel(base_model, args)
        # checkpoint
        checkpoint = ModelCheckpoint(monitor="val_loss", filename=f"best_model_{i}")
        # trainer
        trainer = pl.Trainer(**trainer_params, callbacks=[checkpoint])
        trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="surrogate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dataset_name", type=str, default="tinyimage")
    args = parser.parse_args()
    seed_everything(42)
    main(args)
    print("ok")
