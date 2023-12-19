import os
import argparse
from argparse import Namespace
import sys

sys.path.append("/data/xuth/deep_ipr")

from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchvision.models import vgg16_bn
from torchvision.models import vgg13
from torchvision.models import resnet18
from torchvision.models import densenet121
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from omegaconf import DictConfig


from new2024.util.data_adapter import SplitDataConverter
from new2024.util.data_adapter import defend_attack_split
from new2024.util.metric import ClassificationMetric


class IrrelevantModel(pl.LightningModule):
    def __init__(self, model: nn.Module, conf: DictConfig):
        super().__init__()
        self.model = model
        #
        self.conf = conf
        # loss
        self.train_criterion = CrossEntropyLoss()
        self.val_criterion = CrossEntropyLoss()
        # metric
        self.train_metric = ClassificationMetric(recall=False, precision=False)
        self.val_metric = ClassificationMetric(recall=False, precision=False)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        x, y = batch
        loss = self.train_criterion(_y, y)
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
            self.parameters(),
            lr=self.conf["lr"],
            weight_decay=self.conf["weight_decay"],
        )


def main(args: Namespace):
    conf = OmegaConf.load(args.conf_file)
    project_root_dir = conf["PROJECT_ROOT_DIR"]
    seed_everything(conf["seed"])
    trainer_params = {
        "accelerator": "gpu",
        "devices": [conf["gpu"]],
        "max_epochs": conf["epochs"],  #
        "enable_checkpointing": True,  # True
        "logger": True,  # TensorBoardLogger
        "default_root_dir": os.path.join(
            project_root_dir, "model", conf["model_type"], conf["dataset_name"]
        ),
        "progress_bar_refresh_rate": 1,  # 1
        "num_sanity_val_steps": 0,  # 2
    }
    # prepare data
    train_dataset, dev_dataset, test_dataset = SplitDataConverter().split(conf=conf)
    defend_dataset, attack_dataset = defend_attack_split(train_dataset)
    train_dataloader = DataLoader(
        defend_dataset, batch_size=conf["batch_size"], shuffle=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=conf["batch_size"], shuffle=False
    )
    test_dataset = DataLoader(
        test_dataset, batch_size=conf["batch_size"], shuffle=False
    )
    # prepare model
    for i in range(20):
        if i < 5:
            model = vgg13(weights=True)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feature, conf["classes"])
        elif 5 <= i < 10:
            model = resnet18(weights=True)
            in_feature = model.fc.in_features
            model.fc = nn.Linear(in_feature, conf["classes"])
        elif 10 <= i < 15:
            model = densenet121(weights=True)
            in_feature = model.classifier.in_features
            model.classifier = nn.Linear(in_feature, conf["classes"])
        elif 15 <= i < 20:
            model = mobilenet_v2(weights=True)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feature, conf["classes"])
        model = IrrelevantModel(model, conf)
        # checkpoint
        checkpoint = ModelCheckpoint(
            monitor="val_loss", filename=f"best_model_{i}.ckpt"
        )
        # trainer
        trainer = pl.Trainer(**trainer_params, callbacks=[checkpoint])
        trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_file",
        type=str,
        default="/data/xuth/deep_ipr/new2024/config/cv_irrelevant_tinyimage.yaml",
    )
    args = parser.parse_args()
    main(args)
    print("ok")
