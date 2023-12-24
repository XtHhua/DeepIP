import os
import sys

sys.path.append("/data/xuth/deep_ipr")
import argparse
from argparse import Namespace
from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import vgg16_bn
from torch.utils.data import DataLoader
from torchvision.models import vgg13
from torchvision.models import resnet18
from torchvision.models import densenet121
from torchvision.models import mobilenet_v2
from torch.utils.tensorboard import SummaryWriter

from easydeepip.util.data_adapter import SplitDataConverter
from easydeepip.util.data_adapter import defend_attack_split
from easydeepip.util.seed import seed_everything


def denormalize(image):
    # RGB 三通道像素的均值和方差
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape).to(device)
    for i in range(3):
        img_copy[:, i, :, :] = image[:, i, :, :] * image_data[1, i] + image_data[0, i]
    return img_copy


def normalize(image):
    # RGB 三通道像素的均值和方差
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape).to(device)
    for i in range(3):
        img_copy[:, i, :, :] = (image[:, i, :, :] - image_data[0, i]) / image_data[1, i]
    return img_copy


def PGD(model, image, label):
    label = label.to(device)
    loss_func1 = torch.nn.CrossEntropyLoss()
    image_de = denormalize(deepcopy(image))  # 对输入图像进行反归一化操作
    image_attack = deepcopy(image)
    image_attack = image_attack.to(device)
    # image_attack = torch.tensor(image_attack, requires_grad=True)  # 将输入图像副本转换为可求导的变量
    image_attack = torch.tensor(image_attack).clone().detach().requires_grad_(True)

    alpha = 1 / 256  # 设置步长参数
    epsilon = 4 / 256  # 设置扰动大小的上界

    for iter in range(30):
        # 将输入图像副本设置为可求导的变量
        # image_attack = torch.tensor(image_attack, requires_grad=True)
        image_attack = torch.tensor(image_attack).clone().detach().requires_grad_(True)
        output = model(image_attack)
        # 计算交叉熵损失，前面取反，意为让模型预测分布和label分布差异变大
        loss = -loss_func1(output, label)
        loss.backward()
        grad = image_attack.grad.detach().sign()  # 计算梯度的符号
        image_attack = image_attack.detach()  # 将输入图像副本从计算图中分离
        image_attack = denormalize(image_attack)  # 对输入图像副本进行反归一化操作
        image_attack -= alpha * grad  # 对输入图像副本进行像素更新
        # 计算并限制扰动范围
        eta = torch.clamp(image_attack - image_de, min=-epsilon, max=epsilon)
        image_attack = torch.clamp(image_de + eta, min=0, max=1)
        # 对图像进行归一化操作
        image_attack = normalize(image_attack)
    pred_prob = output.detach()  # 获取模型输出的概率值
    pred = torch.argmax(pred_prob, dim=-1)  # 根据概率值确定预测类别
    acc_num = torch.sum(label == pred)  # 计算正确分类的样本数量
    num = label.shape[0]
    acc = acc_num / num
    acc = acc.data.item()  # 获取准确率的数值表示
    return image_attack.detach(), acc  # 返回对抗样本图像和模型准确率


def train(
    tea_model: nn.Module,
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: Adam,
    scheduler: MultiStepLR,
    verbose: bool = False,
):
    tea_model.eval()
    model.train()
    total_batches = len(data_loader)
    loss_record = []

    for batch_idx, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)
        t_output = tea_model(b_x)
        pred = torch.max(t_output, 1)[1].detach().squeeze()
        b_x_adv, acc = PGD(model, b_x, pred)
        output = model(b_x)
        output_adv = model(b_x_adv)
        loss = loss_fn(output, pred) + loss_fn(output_adv, pred)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
        if batch_idx % 20 == 0 and verbose:
            accuracy = 100 * (output.argmax(1) == b_y).sum().item() / len(b_y)
            print(
                f"loss: {loss:>7f},  accuracy: {accuracy:>0.2f}% [{batch_idx:>5d}/{total_batches:>5d}]"
            )
    scheduler.step()
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.CrossEntropyLoss,
    verbose: bool = False,
):
    total_sample_num = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct_sample_num = 0, 0
    with torch.no_grad():
        for batch_data in data_loader:
            b_x = batch_data[0].to(device)
            b_y = batch_data[1].to(device)
            output = model(b_x)
            test_loss += loss_fn(output, b_y).item()

            correct_sample_num += (
                (output.argmax(1) == b_y).type(torch.float).sum().item()
            )
    #
    test_loss /= num_batches
    #
    test_accuracy = correct_sample_num / total_sample_num
    if verbose:
        print(f"\nloss: {test_loss:>8f}, accuracy: {test_accuracy:>0.2f}%")
    return test_loss, test_accuracy


def fit(
    tea_model: torch.nn.Module,
    model: torch.nn.Module,
    args: Namespace,
    index: int,
    device: torch.device,
):
    tea_model.to(device)
    model.to(device)

    train_dataset, dev_dataset, test_dataset = SplitDataConverter.split(args.dataset)
    defend_dataset, attack_dataset = defend_attack_split(train_dataset)
    train_dataloader = DataLoader(
        defend_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )
    save_dir = args.save_dir
    log_dir = os.path.join(save_dir, f"{index}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    best_acc = 0
    for epoch_id in tqdm(range(args.epochs)):
        train_loss = train(
            tea_model=tea_model,
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=False,
        )
        writer.add_scalar("Loss/Train", train_loss, epoch_id)
        dev_loss, dev_acc = test(
            model=model, data_loader=dev_dataloader, loss_fn=loss_fn, verbose=False
        )
        writer.add_scalar("Loss/Dev", dev_loss, epoch_id)
        writer.add_scalar("Acc/Dev", dev_acc, epoch_id)
        test_loss, test_acc = test(
            model=model, data_loader=test_dataloader, loss_fn=loss_fn, verbose=False
        )
        writer.add_scalar("Loss/Test", test_loss, epoch_id)
        writer.add_scalar("Acc/Test", test_acc, epoch_id)
        if test_acc > best_acc:
            best_acc = test_acc
            model_dir = save_dir.replace("log", "model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(model_dir, f"model_best_{index}.pth")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["vgg13", "resnet18", "densenet121", "mobilenet_v2"],
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--milestones", type=list, default=[60, 120, 160])
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument(
        "--save_dir", type=str, default=f"./log/model_extract_adv/cifar100/"
    )
    args = parser.parse_args()

    seed_everything(2023)

    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    tea_model = vgg16_bn(weights=None)
    num_features = tea_model.classifier[6].in_features
    tea_model.classifier[6] = nn.Linear(num_features, args.classes)
    tea_model.load_state_dict(
        torch.load("./model/source/cifar100/model_best.pth", device)
    )

    model_type = args.model_type
    choices = ["vgg13", "resnet18", "densenet121", "mobilenet_v2"]

    for i in range(5):
        base_index = choices.index(model_type) * 5
        base_index += i
        if base_index < 5:
            model = vgg13(weights="DEFAULT")
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feature, args.classes)
        elif 5 <= base_index < 10:
            model = resnet18(weights="DEFAULT")
            in_feature = model.fc.in_features
            model.fc = nn.Linear(in_feature, args.classes)
        elif 10 <= base_index < 15:
            model = densenet121(weights="DEFAULT")
            in_feature = model.classifier.in_features
            model.classifier = nn.Linear(in_feature, args.classes)
        elif 15 <= base_index < 20:
            model = mobilenet_v2(weights="DEFAULT")
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feature, args.classes)
        fit(
            tea_model=tea_model, model=model, args=args, index=base_index, device=device
        )
