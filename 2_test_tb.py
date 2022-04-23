from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/9715481_b3cb4114ff.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print(img_array.shape)

writer.add_image("test2", img_array, 1, dataformats='HWC')

# y = x
for i in range(100):
    writer.add_scalar("y = x", i, i)

writer.close()
