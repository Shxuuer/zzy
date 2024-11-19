import cv2
import numpy as np
from collections import deque
import time
import utils
import math

colour = ((0, 205, 205), (154, 250, 0), (34, 34, 178), (211, 0, 148), (255, 118, 72), (137, 137, 139))  # 定义矩形颜色
video = "./video/vtest2.mp4"
camera = cv2.VideoCapture(video)
# 参数0表示第一个摄像头
# 判断视频是否打开
if camera.isOpened():
    print("摄像头成功打开")
else:
    print("摄像头未打开")
    exit(0)

# 查看视频相关信息
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(camera.get(cv2.CAP_PROP_FPS))
print("width:", width, "height:", height)

total_floors = 13  # 总层数
floor_height_in_pixels = height / total_floors

# 导出视频
out = cv2.VideoWriter(
    "./video/output1.avi",
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=True
)

grabbed, frame_lwpCV = camera.read()
gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
first_f = np.fft.fft2(gray_lwpCV)

last_gray = gray_lwpCV

pts = [deque(maxlen=30) for _ in range(99999)]
last_time = 0


photo_buffer = []

while True:
    # 读取视频流
    grabbed, frame_lwpCV = camera.read()
    if grabbed is False:
        print("读取结束")
        break

    cv2.imshow("raw", frame_lwpCV)
    # 图像转灰度图
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)

    # 计算相关函数
    imag_f = np.fft.fft2(gray_lwpCV)
    f_prod = imag_f * np.conj(first_f)
    imag_cor = np.real(np.fft.ifft2(f_prod))
    imag_cor = np.fft.fftshift(imag_cor)

    # 计算位移值
    vec_img = imag_cor.ravel()
    maxN_vec_index = np.argsort(vec_img)[-50:]
    maxN_index = np.column_stack(np.unravel_index(maxN_vec_index, imag_cor.shape))
    maxN = vec_img[maxN_vec_index]
    avr_index = maxN.dot(maxN_index) / np.sum(maxN)

    # 图像插值位移
    index_shift = [np.mod(avr_index[0]-height/2, height), np.mod(avr_index[1]-width/2, width)]
    row_sca = (np.array(range(height)) - height/2) / height
    col_sca = (np.array(range(width)) - width/2) / width
    row_shift = np.exp(2j * np.pi * row_sca * index_shift[0])
    col_shift = np.exp(2j * np.pi * col_sca * index_shift[1])
    shift = np.outer(row_shift, col_shift)
    f_shift = imag_f * np.fft.ifftshift(shift)

    imag_shift = np.fft.ifft2(f_shift)
    imag_shift = np.real(np.fft.ifft2(f_shift))
    imag_shift = np.clip(imag_shift, 0, 255)
    imag_shift = cv2.GaussianBlur(imag_shift, (5, 5), 0)
    cv2.imshow("fix", imag_shift / 255)

    # 帧差法
    # 计算差分图
    diff = abs(last_gray - imag_shift)
    last_gray = imag_shift

    # 差分图二值化
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    row_ncare = (np.ceil(np.abs(avr_index[0]-height/2)) + 5).astype(np.int32)
    col_ncare = (np.ceil(np.abs(avr_index[1]-width/2)) + 5).astype(np.int32)
    diff[0:row_ncare, :] = 0
    diff[-row_ncare:, :] = 0
    diff[:, 0:col_ncare] = 0
    diff[:, -col_ncare:0] = 0

    # 显示矩形框
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = diff.astype(np.uint8)
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cont in contours:
        if cv2.contourArea(cont) < 2:
            continue
        count += 1
        rect = cv2.boundingRect(cont)
        x1, y1, w, h = rect
        center = (int((x1 + x1 + w) / 2), int((y1 + y1 + h) / 2))
        pts[count].append(center)
        cv2.circle(frame_lwpCV, center, 10, (0, 0, 255))

        if len(pts[count]) > 2:
            # 计算速度
            dx = pts[count][-1][0] - pts[count][-2][0]
            dy = pts[count][-1][1] - pts[count][-2][1]
            speed = np.sqrt(dx**2 + dy**2)

            # 计算运动方向
            direction = np.arctan2(dy, dx) * 180 / np.pi

            # 检查亮度
            roi = frame_lwpCV[y1:y1+h, x1:x1+w]
            avg_brightness = np.mean(roi)

            # 计算位置
            position_score = (y1 + h / 2) / height  # 位置越靠上，分数越高

                        # 计算预警概率
            probability = (
                0.45 * (speed / 100) +
                0.5 * (abs(direction) / 90) +
                0.05 * position_score
            )
            probability = min(1.0, probability)  # 确保概率不超过1

            # 过滤条件
            if speed > 5 and (direction > 45 or direction < -45) and avg_brightness > 50:
                cv2.rectangle(frame_lwpCV, (x1, y1), (x1 + w, y1 + h), colour[count % 6], 3)
                y = 10 if y1 < 10 else y1
                cv2.putText(frame_lwpCV, "object", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                # cv2.line(frame_lwpCV, (pts[count][-2]), (pts[count][-1]), (255, 0, 0), 4)
                if time.time() - last_time > 5:
                    #计算层数，由于仰视角度，需要修正单楼层高度
                    adjustment_factor = 1 - (y1 / height) * 0.3
                    adjusted_floor_height = floor_height_in_pixels * adjustment_factor
                    object_floor = total_floors - math.floor(y1 / adjusted_floor_height)
                    print(f"object detected at {object_floor} floor.")
                    last_time = time.time()
                    # 将当前帧保存到photo_buffer
                    photo_buffer.append(utils.frame2base64(frame_lwpCV))
                    utils.post(object_floor, 1, probability, photo_buffer)
                    photo_buffer = []
                out.write(frame_lwpCV)

    cv2.imshow("contours", frame_lwpCV)
    out.write(frame_lwpCV)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 释放资源
camera.release()
cv2.destroyAllWindows()