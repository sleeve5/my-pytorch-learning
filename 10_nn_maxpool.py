import torch
from torch import nn
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.MaxPool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        y = self.MaxPool1(x)
        return y


my = MyNN()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images("input", imgs, step)
    print(imgs.shape)
    output = my(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
