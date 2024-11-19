from picamera2 import Picamera2
import cv2
import time
import numpy as np


video = "../video/test_2.mp4"
camera = Picamera2()
camera.start()
time.sleep(1)
# 参数0表示第一个摄像头

"""
# 判断视频是否打开
if camera.isOpened():
    print("摄像头成功打开")
else:
    print("摄像头未打开")


    # 查看视频相关信息
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(camera.get(cv2.CAP_PROP_FPS))
print("width:", width, "height:", height)


# 导出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    '../video/stabilization_test.mp4',
    fourcc, fps, (width, height), isColor=True
)
"""
width, height = 640, 480

# grabbed, frame_lwpCV = camera.read()
frame_lwpCV = camera.capture_array("main")
frame_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_RGB2BGR)

gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
first_f = np.fft.fft2(gray_lwpCV)


last_gray = gray_lwpCV
# last_chan = None

prev_frame_time = 0

while True:

    start_time = time.time()
    
    # 读取视频流
    frame_lwpCV = camera.capture_array("main")
    frame_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_RGB2BGR)

    end_time = time.time()
    fps = 1 / (end_time - prev_frame_time)
    prev_frame_time = end_time

    

    cv2.imshow("Real-Time Camera Feed", frame_lwpCV)
    # 输入时频时循环播放
    '''if frame_lwpCV is None:
        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
        grabbed, frame_lwpCV = camera.read()
        gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
        last_gray = gray_lwpCV
        # last_chan = None'''

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
    maxN_index = np.column_stack(np.unravel_index(
        maxN_vec_index, imag_cor.shape))
    maxN = vec_img[maxN_vec_index]
    avr_index = maxN.dot(maxN_index) / np.sum(maxN)

    # 图像插值位移
    index_shift = [np.mod(avr_index[0]-height/2, height),
                   np.mod(avr_index[1]-width/2, width)]
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
    # cv2.imshow("bi", diff)

    # 显示矩形框
    # 该函数计算一幅图像中目标的轮廓
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = diff.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        diff.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        # 对于矩形区域,只显示大于给定阈值的轮廓,所以一些微小的变化不会显示
        # 对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
        '''if cv2.contourArea(c) < 1500:
            continue'''
        # 该函数计算矩形的边界框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame_lwpCV, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("contours", frame_lwpCV)

    # 导出视频
    # out.write(frame_lwpCV)

    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord("q"):
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
