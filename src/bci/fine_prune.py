import os
import argparse

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torcheeg.model_selection import KFoldGroupbyTrial
from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.model_loader import BCIModelLoader
from util.model_loader import BCIModelLoaderWM
from util.data_adapter import SplitDataConverter
from util.base_model import Conformer
from util.base_model import DeepConvNet
from util.base_model import ShallowConvNet
from util.data_adapter import defend_attack_split


def prune_module(module, prune_rate):
    weight = module.weight.detach().cpu().numpy()
    threshold = np.percentile(np.abs(weight), prune_rate)
    weight[np.abs(weight) < threshold] = 0
    module.weight.data = torch.from_numpy(weight).to(device)


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: Adam,
    epoch_id: int,
) -> float:
    total_batches = len(data_loader)
    #
    model.train()
    print(f"Epoch {epoch_id}\n-------------------------------")
    for batch_idx, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)
        output = model(b_x)
        loss = loss_fn(output, b_y)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        if batch_idx % 20 == 0:
            loss = loss.detach().item()
            accuracy = 100 * (output.argmax(1) == b_y).sum().item() / len(b_y)
            print(f"loss: {loss:>7f},  accuracy: {accuracy:>0.2f}% [{batch_idx:>5d}/{total_batches:>5d}]")
    return loss.cpu().detach().numpy()


def test(model: nn.Module, data_loader: DataLoader):
    model.eval()
    total_sample_num = len(data_loader.dataset)
    total_loss, correct_num = 0, 0
    with torch.no_grad():
        for batch_data in data_loader:
            b_x = batch_data[0].to(device)
            b_y = batch_data[1].to(device)
            output = model(b_x)
            total_loss += F.cross_entropy(output, b_y).item()
            correct_num += (output.argmax(1) == b_y).type(torch.float).sum().item()
    #
    total_loss /= len(data_loader)
    test_accuracy = correct_num / total_sample_num
    return total_loss, round(test_accuracy, 2)


def fine_prune(
    index: int,
    model: Module,
    train_dataset: Dataset,
    attack_dataset: Dataset,
    test_set: Dataset,
    kfold: KFoldGroupbyTrial,
):
    model.to(device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    initial_acc = test(model, train_loader)

    prune_dict = {
        "other_model_name": nn.Conv2d,
    }
    for module in model.modules():
        module_type = type(module).__name__
        if module_type in prune_dict and isinstance(module, prune_dict[module_type]):
            prune_module(module=module, prune_rate=args.prune_ratio)
    prune_acc = test(model, train_loader)
    print(f"After pruning accuracy:{prune_acc}")

    # finetune
    save_dir = f"./model/{args.model_type}/{args.dataset_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for split_idx, (train_attack, dev_attack) in enumerate(kfold.split(attack_dataset)):
        train_loader = DataLoader(dataset=train_attack, batch_size=args.batch_size, shuffle=True, num_workers=4)
        dev_loader = DataLoader(dataset=dev_attack, batch_size=args.batch_size, shuffle=False, num_workers=4)
        #
        best_test_acc = 0
        for epoch_id in range(args.epochs):
            #
            train_loss = train(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch_id=epoch_id,
            )
            print("Loss/Train", train_loss, (split_idx * args.epochs) + epoch_id)
            #
            dev_loss, dev_acc = test(model=model, data_loader=dev_loader)
            print("Loss/Val", dev_loss, (split_idx * args.epochs) + epoch_id)
            #
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f"model_best_{index}.pth"))
        #
        model.load_state_dict(torch.load(os.path.join(save_dir, f"model_best_{index}.pth")))
        _, test_acc = test(model, test_loader)
        print("Acc/Test", test_acc, split_idx + 1)
        print(f"Test Error model_{split_idx}: \n Accuracy: {(100*test_acc):>0.1f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_best_{index}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["ccnn"])
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--label", type=str, default="valence", choices=["valence", "arousal"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prune_ratio", type=float, default=0.5)
    parser.add_argument("--model_type", type=str, default="fine_prune")
    parser.add_argument("--dataset_name", type=str, default="deap")
    args = parser.parse_args()

    #
    seed_everything(2023)
    #
    #
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    train_dataset, _, test_dataset = SplitDataConverter.split(args.dataset_name)
    source_dataset, attack_dataset = defend_attack_split(train_dataset)
    kfold = KFoldGroupbyTrial(n_splits=5, shuffle=True, split_path=f"./data_new/{args.model_name}/train")

    for index in range(10):
        tea_model = BCIModelLoader.load_model(mode="source")
        fine_prune(
            index=index,
            model=tea_model,
            train_dataset=train_dataset,
            attack_dataset=attack_dataset,
            test_set=test_dataset,
            kfold=kfold,
        )
