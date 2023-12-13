import os
import gc
import argparse
from argparse import Namespace
import sys

sys.path.append("/data/xuth/deep_ipr")

import torch
from sklearn.metrics import *
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchvision.models import vgg16_bn
from torch.utils.data import DataLoader

from new2024.util.data_adapter import SplitDataConverter
from new2024.util.data_adapter import defend_attack_split
from new2024.config.config import PROJECT_ROOT_DIR


if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ClassificationMetric(object):  # 记录结果并计算指标
    def __init__(
        self, accuracy=True, recall=True, precision=True, f1=True, average="macro"
    ):
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1 = f1
        self.average = average

        self.preds = []
        self.target = []

    def reset(self):  # 重置结果
        self.preds.clear()
        self.target.clear()
        gc.collect()

    def update(self, preds, target):  # 更新结果
        preds = list(preds.cpu().detach().argmax(1).numpy())
        target = (
            list(target.cpu().detach().argmax(1).numpy())
            if target.dim() > 1
            else list(target.cpu().detach().numpy())
        )
        self.preds += preds
        self.target += target

    def compute(self):  # 计算结果
        metrics = []
        if self.accuracy:
            metrics.append(accuracy_score(self.target, self.preds))
        if self.recall:
            metrics.append(
                recall_score(
                    self.target,
                    self.preds,
                    labels=list(set(self.preds)),
                    average=self.average,
                )
            )
        if self.precision:
            metrics.append(
                precision_score(
                    self.target,
                    self.preds,
                    labels=list(set(self.preds)),
                    average=self.average,
                )
            )
        if self.f1:
            metrics.append(
                f1_score(
                    self.target,
                    self.preds,
                    labels=list(set(self.preds)),
                    average=self.average,
                )
            )
        self.reset()
        return metrics


class CVModel(pl.LightningModule):
    def __init__(self, model: nn.Module, args: Namespace):
        super().__init__()
        self.model = model
        #
        self.args = args
        # loss
        self.train_criterion = CrossEntropyLoss()
        self.val_criterion = CrossEntropyLoss()
        # metric
        self.train_metric = ClassificationMetric(recall=False, precision=False)
        self.val_metric = ClassificationMetric(recall=False, precision=False)
        # log
        self.history = {
            "loss": [],
            "acc": [],
            "f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        loss = self.train_criterion(_y, y)
        self.train_metric.update(_y, y)
        return loss

    def training_epoch_end(self, outs):
        loss = 0.0
        for out in outs:
            loss += out["loss"].cpu().detach().item()
        loss /= len(outs)
        acc, f1 = self.train_metric.compute()
        self.history["loss"].append(loss)
        self.history["acc"].append(acc)
        self.history["f1"].append(f1)

    def validation_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        val_loss = self.val_criterion(_y, y)
        self.val_metric.update(_y, y)
        return val_loss

    def validation_epoch_end(self, outs):
        val_loss = sum(outs).item() / len(outs)
        val_acc, val_f1 = self.val_metric.compute()

        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_f1"].append(val_f1)

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )


def main(args: Namespace):
    trainer_params = {
        "gpus": 1,
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
    base_model = vgg16_bn(weights="DEFAULT")
    model = CVModel(base_model, args)
    # trainer
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="source")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--model_name", type=str, default="vgg16_bn")
    args = parser.parse_args()

    main(args)
    print("ok")
