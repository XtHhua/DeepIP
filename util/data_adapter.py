"""
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2023-12-10 15:04:06
LastEditors: XtHhua
LastEditTime: 2023-12-10 15:22:37
"""
"""
Descripttion: DeepModelIPProtection
version: 1.0
Author: XtHhua
Date: 2023-12-10 15:04:06
LastEditors: XtHhua
LastEditTime: 2023-12-10 15:04:26
"""
import os
import sys

sys.path.append("/data/xuth/deep_ipr")

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT_DIR = "/data/xuth/deep_ipr/easydeepip"


def defend_attack_split(dataset: Dataset):
    """split trainset of cifar series to defend and attach dataset."""
    half_size = len(dataset) // 2
    defend_dataset, attack_dataset = random_split(
        dataset, [half_size, len(dataset) - half_size]
    )
    return defend_dataset, attack_dataset


class TinyImageDatasetFromFolder:
    @staticmethod
    def load_data(selection: str, dir: str, mean: list, std: list):
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        dataset = datasets.ImageFolder(
            root=os.path.join(dir, f"data/tiny-imagenet-200/{selection}"),
            transform=transform,
        )
        return dataset


cifar = {"cifar10": CIFAR10, "cifar100": CIFAR100}


class SplitDataConverter:
    @staticmethod
    def split(dataset_name):
        if dataset_name in ["cifar10", "cifar100"]:
            cifar_mean = [0.4914, 0.4822, 0.4465]
            cifar_std = [0.2023, 0.1994, 0.2010]
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(cifar_mean, cifar_std),
                ]
            )
            train_dataset = cifar[dataset_name](
                root=f"{PROJECT_ROOT_DIR}/data",
                train=True,
                download=True,
                transform=transform,
            )
            test_dataset = cifar[dataset_name](
                root=f"{PROJECT_ROOT_DIR}/data",
                train=False,
                download=True,
                transform=transform,
            )
            train_dataset, dev_dataset = train_test_split(
                train_dataset, test_size=0.1, train_size=0.9, shuffle=True
            )
        elif dataset_name in ["tinyimage"]:
            tiny_mean = [0.5, 0.5, 0.5]
            tiny_std = [0.5, 0.5, 0.5]
            train_dataset = TinyImageDatasetFromFolder.load_data(
                "train", PROJECT_ROOT_DIR, tiny_mean, tiny_std
            )
            dev_dataset = TinyImageDatasetFromFolder.load_data(
                "val", PROJECT_ROOT_DIR, tiny_mean, tiny_std
            )
            test_dataset = TinyImageDatasetFromFolder.load_data(
                "test", PROJECT_ROOT_DIR, tiny_mean, tiny_std
            )
        else:
            raise NotImplementedError
        return train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    # train, dev, test = SplitDataConverter().split("cifar10")
    # print(len(test.classes))
    # defenad, attack = defend_attack_split(train)
    # print(len(defenad), len(attack), len(dev), len(test))
    # train, dev, test = SplitDataConverter().split("cifar100")
    # print(len(test.classes))
    # defenad, attack = defend_attack_split(train)
    # print(len(defenad), len(attack), len(dev), len(test))
    train, dev, test = SplitDataConverter().split("tinyimage")
    print(len(test.classes))

    defenad, attack = defend_attack_split(train)
    print(len(defenad), len(attack), len(dev), len(test))
    print(type(test))
