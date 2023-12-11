import os
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import SGD
from torch.optim import lr_scheduler
from torchvision.models import vgg16_bn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from new2024.util import seed
from new2024.util import data_adapter
from new2024.config.config import PROJECT_ROOT_DIR


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: SGD,
    scheduler: lr_scheduler.CosineAnnealingLR,
    device: torch.device,
    verbose: bool = False,
):
    model.train()
    total_batches = len(data_loader)
    loss_record = []

    for batch_idx, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)
        output = model(b_x)
        loss = loss_fn(output, b_y)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
        if batch_idx % 20 == 0 and verbose:
            accuracy = 100 * (output.argmax(1) == b_y).sum().item() / len(b_y)
            print(
                f"loss: {loss:>7f},  accuracy: {accuracy:>0.2f}% [{batch_idx:>5d}/{total_batches:>5d}]"
            )
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.CrossEntropyLoss,
    device: torch.device,
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
    model: torch.nn.Module,
    args: argparse.ArgumentParser,
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
):
    model.to(device)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    save_dir = f"{PROJECT_ROOT_DIR}/model/source/{args.dataset}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    best_acc = 0
    for epoch_id in tqdm(range(args.epochs)):
        train_loss = train(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            verbose=False,
        )
        writer.add_scalar("Loss/Train", train_loss, epoch_id)
        test_loss, test_acc = test(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            verbose=False,
        )
        writer.add_scalar("Loss/Test", test_loss, epoch_id)
        writer.add_scalar("Acc/Test", test_acc, epoch_id)
    writer.close()
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    args = parser.parse_args()

    seed.seed_everything()

    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    defend_dataset, attack_dataset = data_adapter.split_cifar_train(
        dataset_name=args.dataset
    )
    test_dataset = data_adapter.load_test_dataset(dataset_name=args.dataset)

    model = vgg16_bn(weights=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, len(test_dataset.classes))

    fit(
        model=model,
        args=args,
        train_dataset=defend_dataset,
        test_dataset=test_dataset,
        device=device,
    )

    # # performance 0.97
    # model.load_state_dict(
    #     torch.load(f"./model/source_train/model_best.pth", map_location=device)
    # )
    # train_dataset, _ = split_dataset.split_cifar100_train()
    # trigger_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=10,
    #     shuffle=False,
    # )
    # acc = utils.test(model=model, dataloader=trigger_loader, device=device)
    # print(acc)
