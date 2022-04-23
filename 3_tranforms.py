from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "data/train/ants_image/9715481_b3cb4114ff.jpg"
img = Image.open(img_path)

# use transforms
tensor_trans = transforms.ToTensor()    # a tool
tensor_img = tensor_trans(img)  # use tool

writer = SummaryWriter("logs")

writer.add_image("tensor_img", tensor_img)

writer.close()
