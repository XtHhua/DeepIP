import os
import argparse
from tqdm import tqdm

import torch
import torchvision
import torch.optim as optim
from torch import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from new2024.util import seed
from new2024.util import data_adapter
from new2024.util.model_loader import CVModel
from new2024.config.config import PROJECT_ROOT_DIR


def train_student_model(
    index: int,
    teacher: Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    args: argparse.ArgumentParser,
):
    teacher = teacher.to(device)
    teacher.eval()

    accu_best = 0

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = torchvision.models.vgg16_bn(weights=True)
    in_feature = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_feature, len(test_dataset.classes))

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    save_dir = f"{PROJECT_ROOT_DIR}/model/surrogate/{args.dataset}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    for epoch in tqdm(range(args.epochs)):
        model.to(device)
        model.train()
        train_loss = []
        for batch_idx, batch_data in enumerate(train_loader):
            b_x = batch_data[0].type(torch.FloatTensor).to(device)
            b_y = batch_data[1].long().to(device)
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].detach().squeeze()
            output = model(b_x)
            loss = loss_func(output, pred)
            train_loss.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss = sum(train_loss) / len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        model.eval()
        num = 0
        total_num = 0
        test_loss = []
        for batch_idx, batch_data in enumerate(test_loader):
            b_x = batch_data[0].type(torch.FloatTensor).to(device)
            b_y = batch_data[1].long().to(device)
            output = model(b_x)
            pred = torch.max(output, 1)[1].detach().squeeze()
            loss = loss_func(output, pred)
            test_loss.append(loss.detach().item())
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]

        test_loss = sum(test_loss) / len(test_loader)
        writer.add_scalar("Loss/Test", test_loss, epoch)

        accu1 = num / total_num
        writer.add_scalar("Acc/Test", accu1, epoch)
        # print("Epoch:", epoch + 1, "accuracy:", accu1)

        if accu1 > accu_best:
            accu_best = accu1
            torch.save(
                model.state_dict(), os.path.join(save_dir, f"model_{index}_best.pth")
            )

    return accu_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    args = parser.parse_args()

    seed.seed_everything()

    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    cv_model_loader = CVModel(dataset_name=args.dataset, device=device)
    teacher = cv_model_loader.load_model(mode="source")

    defend_dataset, attack_dataset = data_adapter.split_cifar_train(
        dataset_name=args.dataset
    )
    test_dataset = data_adapter.load_test_dataset(dataset_name=args.dataset)

    for i in range(10):
        accu = train_student_model(
            index=i,
            teacher=teacher,
            train_dataset=defend_dataset,
            test_dataset=test_dataset,
            device=device,
            args=args,
        )
