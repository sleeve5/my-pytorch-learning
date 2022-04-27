# author: Yin Xinyu
import torch
import torchvision.transforms
from PIL import Image

image_path = "dog.png"
image = Image.open(image_path)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)

model = torch.load("nn_train.pth")

image = torch.reshape(image, (1, 3, 32, 32))
image = image.cuda()
model.eval()
with torch.no_grad():
    output = model(image)

print(output.argmax(1))
