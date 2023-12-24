import os
import argparse
from copy import deepcopy
import sys

sys.path.append("/data/xuth/deep_ipr")

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision.models import vgg16_bn
from torch.utils.data import DataLoader

from easydeepip.util.data_adapter import SplitDataConverter
from easydeepip.util.data_adapter import defend_attack_split


class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


def find_smallest_neuron(hook_list, prune_list):
    activation_list = []
    for j in range(len(hook_list)):
        activation = hook_list[j].output
        for i in range(activation.shape[1]):
            activation_channel = torch.mean(torch.abs(activation[:, i, :, :]))
            activation_list.append(activation_channel)

    activation_list1 = []
    activation_list2 = []

    for n, data in enumerate(activation_list):
        if n in prune_list:
            pass
        else:
            activation_list1.append(n)
            activation_list2.append(data)

    activation_list2 = torch.tensor(activation_list2)
    prune_num = torch.argmin(activation_list2)
    prune_idx = activation_list1[prune_num]

    return prune_idx


def finetune_step(model, dataloader, criterion):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) * inputs.shape[0] >= 2056:
            break


def value(model, dataloader):
    model.eval()
    num = 0
    total_num = 0
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.to(device), y.to(device)
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()
        num += (pred == b_y).sum().item()
        total_num += pred.shape[0]

    accu = num / total_num
    return accu


def run_model(model, dataloader):
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.to(device), y.to(device)
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()


def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx


def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0


def fine_pruning(model, train_loader, test_loader):
    model.to(device)
    module_list = []
    neuron_num = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_list.append(module)
            neuron_num.append(module.out_channels)

    neuron_num = np.array(neuron_num)
    max_id = np.sum(neuron_num)

    neuron_list = []
    mask_list = []
    for i in range(neuron_num.shape[0]):
        neurons = list(range(neuron_num[i]))
        neuron_list.append(neurons)
        prune_filter = prune.identity(module_list[i], "weight")
        mask_list.append(prune_filter)

    prune_list = []
    init_val = value(model, test_loader)
    acc = []
    length = deepcopy(len(neuron_list))
    total_length = 0
    for i in range(length):
        total_length += len(neuron_list[i])
    print("Total number of neurons is", total_length)
    flag = True
    for i in range(int(np.floor(args.prune_amount * total_length))):
        if flag:
            hook_list = []
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    hook_list.append(FeatureHook(module))
            run_model(model, train_loader)
        idx = find_smallest_neuron(hook_list, prune_list)
        prune_list.append(idx)
        prune_neuron(mask_list, idx, neuron_num)
        if i % 50 == 0:
            finetune_step(model, train_loader, criterion=torch.nn.CrossEntropyLoss())
        if i % 50 == 0:
            new_val = value(model, test_loader)
            print("neuron remove:", i, "init_value:", init_val, "new_value:", new_val)
            acc.append([i, new_val])

        if (
            np.floor(20 * i / total_length) - np.floor(20 * (i - 1) / total_length)
        ) == 1:
            iter = int(np.floor(20 * i / total_length))
            save_dir = args.save_dir
            save_dir = save_dir.replace("log", "model")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for hook in hook_list:
                hook.close()
            torch.save(model, os.path.join(save_dir, f"model_best_{str(iter)}.pth"))
            print(f"neuron remove: {i}, Saving model! Model number is:{iter}")
            flag = True
        else:
            flag = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--save_dir", type=str, default=f"./log/fine_prune/cifar100/")
    parser.add_argument("--prune_amount", type=float, default=0.80)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    tea_model = vgg16_bn(weights=None)
    num_features = tea_model.classifier[6].in_features
    tea_model.classifier[6] = nn.Linear(num_features, args.classes)
    tea_model.load_state_dict(
        torch.load("./model/source/cifar100/model_best.pth", device)
    )

    print("model_load")
    train_dataset, dev_dataset, test_dataset = SplitDataConverter.split(args.dataset)
    defend_dataset, attack_dataset = defend_attack_split(train_dataset)

    train_dataloader = DataLoader(
        defend_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    mem = fine_pruning(tea_model, train_dataloader, test_dataloader)

    # python pruning.py
