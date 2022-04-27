# author: Yin Xinyu
import torch
from torch import nn


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
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        y = self.model(x)
        return y


if __name__ == '__main__':
    nn = Mynn()
    input = torch.ones((64, 3, 32, 32))
    output = nn(input)
    print(output.shape)
