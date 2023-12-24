import os

import torch
from torch import device
from torch.nn.modules import Module
from torchvision import models

PROJECT_ROOT_DIR = "/data/xuth/deep_ipr/easydeepip"


class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


class CVModelLoader:
    def __init__(self, dataset_name: str, device: device = None):
        self.dataset_name = dataset_name
        self.device = device if device else torch.device("cpu")
        self.base_dir = PROJECT_ROOT_DIR + "/model"
        if self.dataset_name == "cifar100":
            self.classes = 100
        elif self.dataset_name == "cifar10":
            self.classes = 10
        elif self.dataset_name == "tinyimage":
            self.classes = 200
        else:
            raise NotImplementedError

    def load_model(self, mode: str, index: int = 0) -> Module:
        model_dir = os.path.join(PROJECT_ROOT_DIR, "model")
        if mode == "source":
            model = models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir, f"{mode}", f"{self.dataset_name}", "model_best.pth"
                    ),
                    self.device,
                )
            )
        elif mode == ["surrogate", "finetune", "fineprune", "transfer_learning"]:
            model = models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir,
                        f"{mode}",
                        f"{self.dataset_name}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        elif mode in [
            "irrelevant",
            "model_extracl_l",
            "model_extract_p",
            "model_extract_adv",
        ]:
            if index < 5:
                model = models.vgg13(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            elif 5 <= index < 10:
                model = models.resnet18(weights=None)
                in_feature = model.fc.in_features
                model.fc = torch.nn.Linear(in_feature, self.classes)
            elif 10 <= index < 15:
                model = models.densenet121(weights=None)
                in_feature = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_feature, self.classes)
            elif 15 <= index < 20:
                model = models.mobilenet_v2(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir,
                        f"{mode}",
                        f"{self.dataset_name}",
                        f"model_best_{index}.pth",
                    ),
                    self.device,
                )
            )
        return model


class NLPModel:
    def __init__(self) -> None:
        pass


class BCIModel:
    def __init__(self) -> None:
        pass
