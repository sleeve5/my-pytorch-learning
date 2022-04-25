import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        y = self.relu1(x)
        return y


my = Mynn()
writer = SummaryWriter("logs")

step = 0
for data in dataset:
    imgs, targets = data
    # imgs = torch.reshape(imgs, (-1, 3, 32, 32))
    writer.add_images("input", imgs, global_step=step)  # !
    output = my(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()
