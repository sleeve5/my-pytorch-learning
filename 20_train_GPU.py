# author: Yin Xinyu

import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        y = self.model(x)
        return y


# 数据集准备
train_data = torchvision.datasets.CIFAR10("dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度{}".format(train_data_size))
print("测试集长度{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建模型
nn_train = Mynn()

if torch.cuda.is_available():
    nn_train = nn_train.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(nn_train.parameters(), lr=learning_rate)

# 设置模型参数

# 记录训练次数
total_train_step = 0

# 记录测试次数
total_test_step = 0

# 添加可视化界面
writer = SummaryWriter("logs/train")

# 设置训练轮
epoch = 10
time_start = time.time()

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))

    # 开始训练
    nn_train.train()
    for data in train_dataloader:
        imgs, targets = data

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        output = nn_train(imgs)
        loss = loss_fn(output, targets)

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 训练过程记录
        total_train_step += 1
        if total_train_step % 100 == 0:
            time_end = time.time()
            print("用时{}".format(time_end - time_start))
            print("训练次数{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 开始测试
    # 训练集上loss计算
    nn_train.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            output = nn_train(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item()

            # 正确率计算
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体数据集上Loss:{}".format(total_test_loss))
    print("整体数据集上正确率:{}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(nn_train, "models/nn_train{}.pth".format(i))
    print("模型已保存")

torch.save(nn_train, "nn_train.pyh")

writer.close()
