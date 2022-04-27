# author: Yin Xinyu
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("dataset/image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_t = torchvision.models.vgg16(pretrained=True)
vgg16_f = torchvision.models.vgg16(pretrained=False)

train_data = torchvision.datasets.CIFAR10("dataset/cifar10", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

vgg16_t.classifier.add_module('add_Linear', nn.Linear(1000, 10))
# print(vgg16_t)
print(vgg16_f)
vgg16_f.classifier[6] = nn.Linear(4096, 10)
print(vgg16_f)
