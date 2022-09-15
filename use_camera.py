import cv2
from process_frame import process_frames
from torchvision.models import MobileNetV2
import torch

# 加载模型
model = MobileNetV2()
model.load_state_dict(torch.load('./mobilenet_v2.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 获取摄像头，传入0表示默认摄像头，Mac系统应改为1
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        break
    # 处理帧数
    frame = process_frames(model, frame)
    # 展示处理后的三通道
    cv2.imshow('my_window', frame)
    if cv2.waitKey(1) in [ord('q'), 27]: # 按键盘上的q或esc退出(在英文输入法上)
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()
