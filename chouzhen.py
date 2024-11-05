import cv2  
import math  
  
# 输入和输出视频文件路径  
input_video_path = './IMG_4550.MOV'  
output_video_path = './IMG_4550_2.MOV'  
  
# 目标FPS（例如，将原始FPS减半）  
target_fps = 2  # 假设原始FPS是30，这里设置为15  
  
# 打开视频文件  
cap = cv2.VideoCapture(input_video_path)  
  
# 获取视频的原始FPS和帧尺寸  
original_fps = cap.get(cv2.CAP_PROP_FPS)  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
  
# 计算新的帧间隔（基于原始FPS和目标FPS的比例）  
frame_interval = math.ceil(original_fps / target_fps)  
  
# 定义视频编解码器并创建VideoWriter对象  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编解码器  
out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))  
  
frame_idx = 0  
selected_frame_idx = 0  
while cap.isOpened():  
    ret, frame = cap.read()  
      
    if not ret:  
        break  
      
    # 检查是否应该选取当前帧  
    if selected_frame_idx == frame_idx:  
        out.write(frame)  # 写入选取的视频帧  
        selected_frame_idx += frame_interval  # 更新下一个要选取的帧索引  
      
    frame_idx += 1  
  
# 释放资源  
cap.release()  
out.release()