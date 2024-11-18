import base64
import requests
import cv2
# 
def post(floor, level, probability, photos):
    BASE_URL = "http://localhost:3000"
    MECHINE_NAME = "测试"
    url = BASE_URL + "/raspberry/submit"
    body = {
        "location": MECHINE_NAME,
        "floor": floor,
        "level": level,
        "probability": probability,
        "photos": photos
    }
    response = requests.post(url, json=body)
    print(response.text)

def frame2base64(frame):
    # 将图片转为base64，格式"data:image/png;base64,xxxxxxx"
    _, buffer = cv2.imencode('.png', frame)
    return "data:image/png;base64," + base64.b64encode(buffer).decode()
