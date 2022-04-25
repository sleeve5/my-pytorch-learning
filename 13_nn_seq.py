import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()

        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.MaxPool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.MaxPool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.MaxPool3 = MaxPool2d(2)
        # self.Flatten = Flatten()
        # self.Linear1 = Linear(1024, 64)
        # self.Linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.MaxPool1(x)
        # x = self.conv2(x)
        # x = self.MaxPool2(x)
        # x = self.conv3(x)
        # x = self.MaxPool3(x)
        # x = self.Flatten(x)
        # x = self.Linear1(x)
        # x = self.Linear2(x)
        x = self.model1(x)
        return x


nn = Mynn()
input = torch.ones((64, 3, 32, 32))
output = nn(input)

writer = SummaryWriter("logs/seq")
writer.add_graph(nn, input)
writer.close()
