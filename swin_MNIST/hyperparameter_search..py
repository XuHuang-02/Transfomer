import os
import argparse
import torch
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_mnist as create_model
from utils import read_split_data, train_one_epoch, evaluate


def objective(trial, args, train_loader, val_loader, device):
    # 使用 Optuna 采样超参数
    patch_size = trial.suggest_int('patch_size', 1, 4)
    window_size = trial.suggest_int('window_size', 2, 4)
    embed_dim = trial.suggest_categorical('embed_dim', [48, 96, 192])
    depths_str = trial.suggest_categorical('depths', [(2, 2, 6, 2), (2, 2, 18, 2)])
    depths = eval(depths_str)
    # 创建模型
    model = create_model(num_classes=args.num_classes,
                         patch_size=patch_size,
                         window_size=window_size,
                         embed_dim=embed_dim,
                         depths=depths).to(device)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)

    # 训练和验证
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 向 Optuna 报告结果
        trial.report(val_loss, epoch)

        # 如果在训练过程中验证集上的损失没有提升，可以提前终止
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss  # 也可以返回 `-val_acc` 来最大化准确率


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 读取数据集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 设置图像转换
    img_size = 28
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    }

    # 创建数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 设置数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    # 创建 Optuna study
    study = optuna.create_study(direction="minimize")  # 可以是 "maximize" 如果是优化准确率
    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader, device),
                   n_trials=50)

    # 输出最优超参数
    print(f"Best trial: {study.best_trial.params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default="./data/MNIST")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    main(args)
