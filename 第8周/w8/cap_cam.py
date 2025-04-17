import cv2
import base64
import time

from volcenginesdkarkruntime import Ark

from ark_key_model import KEY as API_KEY
from ark_key_model import MODEL as ARK_MODEL

#pip install sounddevice
import sounddevice as sd
import soundfile as sf

# 初始化一个Client对象，提供API Key
client = Ark(api_key = API_KEY)

def play_sound(filename):
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    #等待播放完成
    status = sd.wait()

def get_cap_from_ark(image_base64, image_format, prompt):
    response = client.chat.completions.create(
      model = ARK_MODEL,
      messages = [
        {
          "role": "user",
          "content": [
            {
             "type": "text",
             "text": f"{prompt}"
        },
        {
          "type": "image_url",
          "image_url": {
          "url":  f"data:image/{image_format};base64,{image_base64}"
          },
        },
      ],
     }
    ],
    )
    return response.choices[0].message.content

def capture_single_frame():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取画面")
        return None

    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return frame_base64

def understand_frame(prompt, welcome, intervals = 60):
    while True:
        frame_base64 = capture_single_frame()
        cap = get_cap_from_ark(frame_base64, 'jpg', prompt)
        print(cap)
        if cap == '1':
            play_sound(welcome)
        else:
            print("无发现。。。。")
        time.sleep(intervals)#控制采样频率

def main():
    prompt = '请分析图片中是否有人出现。如果有，请返回1，否则返回0。'
    welcome = 'qq.wav'
    intervals = 60
    understand_frame(prompt, welcome, intervals = intervals)

if __name__ == '__main__':
    main()