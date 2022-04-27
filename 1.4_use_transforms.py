from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("data/train/ants_image/0013035.jpg")

# totensor
tran_totensor = transforms.ToTensor()
img_tensor = tran_totensor(img)
writer.add_image("tensor", img_tensor)

# normalize
# print(img_tensor[0][0][0])
tran_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = tran_norm(img_tensor)
# print(img_norm[0][0][0])
writer.add_image("norm", img_norm)

# resize
# print(img.size)
tran_resize = transforms.Resize((512, 512))
img_resize = tran_resize(img)
img_resize = tran_totensor(img_resize)
writer.add_image("resize", img_resize, 0)
# print(img_resize)

# ComposeResize
tran_resize_2 = transforms.Resize(512)
tran_compose = transforms.Compose([tran_resize_2, tran_totensor])
img_resize_2 = tran_compose(img)
writer.add_image("resize 2", img_resize_2, 1)

# !RandomCrop
tran_random = transforms.RandomCrop(512)
tran_compose_2 = transforms.Compose([tran_random, tran_totensor])
for i in range(10):
    img_crop = tran_compose_2(img)
    writer.add_image("Random", img_crop, 3)



writer.close()
