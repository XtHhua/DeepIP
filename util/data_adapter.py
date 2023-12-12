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
import sys

sys.path.append("/data/xuth/deep_ipr")

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split


from new2024.config.config import PROJECT_ROOT_DIR
from new2024.config.config import CIFAR_MEAN
from new2024.config.config import CIFAR_STD

CIFAR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)


def split_cifar_train(dataset_name: str):
    """split trainset of cifar series to defend and attach dataset."""

    if dataset_name == "cifar100":
        cifar_dataset = CIFAR100(
            root=f"{PROJECT_ROOT_DIR}/data",
            train=True,
            download=True,
            transform=CIFAR_TRANSFORM,
        )
    elif dataset_name == "cifar":
        cifar_dataset = CIFAR10(
            root=f"{PROJECT_ROOT_DIR}/data",
            train=True,
            download=True,
            transform=CIFAR_TRANSFORM,
        )

    half_size = len(cifar_dataset) // 2
    defend_dataset, attack_dataset = random_split(
        cifar_dataset, [half_size, len(cifar_dataset) - half_size]
    )
    return defend_dataset, attack_dataset


def load_test_dataset(dataset_name: str):
    if dataset_name == "cifar10":
        test_dataset = CIFAR10(
            root=f"{PROJECT_ROOT_DIR}/data",
            train=False,
            download=True,
            transform=CIFAR_TRANSFORM,
        )
    elif dataset_name == "cifar100":
        test_dataset = CIFAR100(
            root=f"{PROJECT_ROOT_DIR}/data",
            train=False,
            download=True,
            transform=CIFAR_TRANSFORM,
        )
    return test_dataset


if __name__ == "__main__":
    load_test_dataset(dataset_name="cifar100")
