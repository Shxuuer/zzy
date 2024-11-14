import numpy as np
import cv2
from line import Draw
from collections import deque
import line

colour = ((0, 205, 205), (154, 250, 0), (34, 34, 178), (211, 0, 148), (255, 118, 72), (137, 137, 139))  # 定义矩形颜色

cap = cv2.VideoCapture(r"D:\\PycharmProject\\gkpw\\data\\vtest3.mp4")  # 参数为0是打开摄像头，文件名是打开视频

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

width1 = width
height1 = height

# 如果视频是横屏屏的，交换宽度和高度
if width > height:
    width1, height1 = height, width

fgbg = cv2.createBackgroundSubtractorKNN(history=7, dist2Threshold=1000, detectShadows=False)  # 混合高斯背景建模算法
# KNN系列主要针对的是比较高层一些的楼，用聚类的特点将背景进行检测
# MOG2主要针对底层楼 分为2-5个高斯背景进行检测

# history：用于训练背景的帧数，默认为500帧，原论文中作者提出7帧效果最优，如果不手动设置learningRate，history就被用于计算当前的learningRate，此时history越大，learningRate越小，背景更新越慢；
# varThreshold/dist2Threshold：方差阈值，用于判断当前像素是前景还是背景。一般默认16，如果光照变化明显，如阳光下的水面，建议设为25,36，具体去试一下也不是很麻烦，值越大，灵敏度越低；
# 低灵敏度好出：对于视频中出现的噪音点会忽视，但是针对视频中高速运动的物体可以检测

# detectShadows：是否检测影子，设为true为检测，false为不检测，检测影子会增加程序时间复杂度，如无特殊要求，建议设为false；

# 初始化一个包含99999个deque对象的列表，每个deque对象的最大长度为30
pts = [deque(maxlen=30) for _ in range(99999)]

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# print("aaaafgfffffffffff", fps)
# print("aaaaaaaaaaaaaaaaa", width, height)

# 创建视频解码器，使用MJPG编码
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 解码器
# 初始化VideoWriter对象，用于将处理后的帧写入到指定路径的视频文件中
out = cv2.VideoWriter("D:\\PycharmProject\\gkpw\\output.avi", fourcc, fps, (width1, height1))

# 提取视频的频率，每1帧提取一个
frameFrequency = 1
times = 0
time_list = []

import math


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (
                point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (
                point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (
                point_1[1] - point_2[1]))
    # A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    # C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    return B


print(cal_ang((0, 0), (1, 1), (0, 1)))

while cap.isOpened():
    times += 1  # 增加计数器，用于跟踪处理的帧数
    # 读取视频流的一帧，ret表示是否成功读取，frame是读取到的图像数据
    ret, frame = cap.read()
    if not ret:
        break
    if width > height:
        # 旋转图像90度
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    Draw(frame)  # 调用Draw函数对读取到的图像进行绘制处理

    # print(frame)
    if frame is not None:
        frame[0:60, 2:280] = [0, 0, 0]  # 对左上角的时间进行覆盖，避免画面出现噪音点
        frame[0:60, 790:970] = [0, 0, 0]

        # 应用背景减除法获取前景掩码
        fgmask = fgbg.apply(frame)
        # 对帧进行模糊处理，减少噪声
        element = cv2.blur(frame, (3, 3))
        # 应用方框滤波器，进一步平滑图像
        element = cv2.boxFilter(element, -1, (2, 2), normalize=False)
        # 获取形态学椭圆结构元素，用于后续的形态学处理
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 形态学去噪
        # 使用开运算去除掩码中的噪声
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪
        # 寻找前景物体的轮廓
        _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景

        count = 0
        for cont in contours:
            Area = cv2.contourArea(cont)  # 计算轮廓面积
            if Area < 2:  # 过滤面积小于1的形状
                continue
            count += 1  # 计数加一
            print("{}-prospect:{}".format(count, Area), end="  ")  # 打印出每个前景的面积
            rect = cv2.boundingRect(cont)  # 提取矩形坐标
            print("x:{} y:{}".format(rect[0], rect[1]))  # 打印坐标
            x1 = rect[0]
            x2 = rect[0] + rect[2]
            y1 = rect[1]
            y2 = rect[1] + rect[3]
            center = ((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            pts[count].append(center)
            print("中心点坐标:", list(center))
            pts[count].append(center)
            cv2.circle(frame, center, 10, (0, 0, 255))

            for j in range(1, len(pts[count])):

                # 计算当前点与前一点之间的斜率，避免除以零通过添加1e-1进行平滑处理
                xielv = (pts[count][j][0] - pts[count][j - 1][0]) / (pts[count][j][1] - pts[count][j - 1][1] + 1e-1)

                # 调试输出计算得到的斜率值
                print("aaaaaaaaaaaaaasasdasdasdasdasd", xielv)

                if xielv == 0:
                    print("xielv", xielv)
                    # print("hudu", radin)
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                                  colour[count % 6],
                                  3)  # 原图上绘制矩形

                    cv2.rectangle(fgmask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                                  (0xff, 0xff, 0xff),
                                  3)  # 黑白前景上绘制矩形
                    y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
                    cv2.putText(frame, "object", (rect[0], y), cv2.CHAIN_APPROX_NONE, 2, (0, 255, 0),
                                2)  # 在前景上写上编号
                    cv2.line(frame, (pts[count][j - 1]), (pts[count][j]), (255, 0, 0), 4)
                    print("警报！高空抛物出现,出现的帧数为:", times)
                    time_list.append(times)  # 将获取的帧数进行保存
                    out.write(frame)
            print("视频总帧数", times)

        cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)  # 显示总数
        cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        # print("帧数列表:", time_list)
        cv2.imshow('frame', frame)  # 在原图上标注
        cv2.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景

        print("----------------------------")

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800, 600)

        cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame2", 800, 600)

        cv2.imshow('frame', frame)  # 在原图上标注
        cv2.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景

    if cv2.waitKey(1) & 0XFF == ord("q"):
        out.release()  # 释放文件
        cap.release()
        cv2.destroyAllWindows()
        break
