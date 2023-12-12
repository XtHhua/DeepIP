import torch
from torch import device
from torch.nn.modules import Module
from torchvision import models


from new2024.config.config import PROJECT_ROOT_DIR


class CVModel:
    def __init__(self, dataset_name: str, device: device):
        self.dataset_name = dataset_name
        self.device = device
        self.base_dir = PROJECT_ROOT_DIR + "/model"
        if self.dataset_name == "cifar100":
            self.classes = 100
        elif self.dataset_name == "cifar10":
            self.classes = 10

    def load_model(self, mode: str, index: int = 0) -> Module:
        if mode == "source":
            model = models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    f"{self.base_dir}/{mode}/{self.dataset_name}/model_best.pth"
                ),
                self.device,
            )
        elif mode == ["surrogate", "finetune", "fineprune", "transfer_learning"]:
            model = models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, self.classes)
            model.load_state_dict(
                torch.load(
                    f"{self.base_dir}/{mode}/{self.dataset_name}/model_{index}_best.pth"
                ),
                self.device,
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
                    f"{self.base_dir}/{mode}/{self.dataset_name}/model_{index}_best.pth"
                ),
                self.device,
            )
        return model


class NLPModel:
    def __init__(self) -> None:
        pass


class BCIModel:
    def __init__(self) -> None:
        pass
