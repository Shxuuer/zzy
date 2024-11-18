import cv2

def camera_read(camera):
    if camera.isOpened():
        print("摄像头成功打开")
    else:
        print("摄像头未打开")

    # 查看视频相关信息
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    print("width:", width, "height:", height)
