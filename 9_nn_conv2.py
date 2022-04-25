import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Mynn(nn.Module):

    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


nn = Mynn()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = nn(imgs)
    print(imgs.shape)

    writer.add_images("input", imgs, step, dataformats="NCHW")
    output = torch.reshape(output, (-1, 3, 30, 30))
    print(output.shape)
    writer.add_images("output", output, step, dataformats="NCHW")   # !

    step = step + 1

writer.close()
