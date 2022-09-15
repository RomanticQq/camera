import numpy as np

import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.models import MobileNetV2
import time

model = MobileNetV2()
start_time = time.time()
img_path = 'img.png'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = np.transpose(img, (2, 0, 1)).astype(dtype=np.float32)
img = np.expand_dims(img, axis=0)
img = img / 255
img = torch.from_numpy(img)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = img.to(device)
model = model.to(device)
pre = model(img)
ori_img = cv2.imread(img_path)
classes = torch.max(pre, dim=1)[1]

text1 = "{}".format(classes.item())
image = cv2.putText(ori_img, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 20, 60), 1)
end_time = time.time()
FPS = 1 / (end_time - start_time)
image = cv2.putText(ori_img, str(int(FPS)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 20, 60), 1)


def look_img(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


look_img(image)


