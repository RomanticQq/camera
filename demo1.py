import numpy as np

import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.models import MobileNetV2
from classes_name import class_name
import torch.nn as nn
import time

model = MobileNetV2()
model.load_state_dict(torch.load('./mobilenet_v2.pth'))
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
top = 5
pre = pre[0]
softmax = nn.Softmax(dim=0)
pre_softmax = softmax(pre)
pre_value, pre_index = torch.sort(pre_softmax, descending=True)  # descending true是降序 false是升序
pre_index = pre_index[:top]
pre_value = pre_value[:top]
for i in range(top):
    pre_class_index = pre_index[i].item()
    text1 = "{:<10s}: {:.4f}".format(class_name[pre_class_index][0], pre_value[i].item())
    image = cv2.putText(ori_img, text1, (10, 50 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 20, 60), 1)
end_time = time.time()
FPS = 1 / (end_time - start_time)
fps = "FPS: {}".format(str(int(FPS)))
image = cv2.putText(ori_img, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 20, 60), 1)


def look_img(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


look_img(image)


