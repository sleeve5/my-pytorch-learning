import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        y = self.linear1(x)
        return y


my = Mynn()

step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = my(output)
    print(output.shape)
