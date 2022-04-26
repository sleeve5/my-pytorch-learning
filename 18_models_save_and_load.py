# author: Yin Xinyu
import torch
import torchvision.models

vgg16 = torchvision.models.vgg16(pretrained=False)

# method 1
torch.save(vgg16, "vgg16_method_1.pth")                 # save model itself by pth

model1 = torch.load("vgg16_method_1.pth")               # load model
print(model1)

# method 2
torch.save(vgg16.state_dict(), "vgg_method_2.pth")      # save model para by dict

model2 = torchvision.models.vgg16(pretrained=False)     # load model
model2.load_state_dict(torch.load("vgg_method2.pth"))
print(model2)
