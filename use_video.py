import cv2
from process_frame import process_frames
from tqdm import tqdm
from torchvision.models import MobileNetV2
import torch

# 加载模型
model = MobileNetV2()
model.load_state_dict(torch.load('./mobilenet_v2.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def generate_video(input_path):
    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                try:
                    frame = process_frames(model, frame)
                except:
                    print('error')
                    pass

                if success:
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

        except:
            print("中途中断")
            pass
        cv2.destroyAllWindows()
        out.release()
        cap.release()
        print("视频已保存", output_path)


generate_video('20220915_111642.mp4')
