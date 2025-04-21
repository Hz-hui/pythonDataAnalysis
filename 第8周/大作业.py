import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
import base64
import os
from volcenginesdkarkruntime import Ark
import time
from datetime import datetime
import logging

"""
1. 实现一个类装饰器ImageHSV,
其计算图像进行预处理前图像的大小、
亮度和饱和度，并输出到特定的日志文件中。
可以使用OpenCV库来计算图像的大小、
亮度和饱和度,
如可将图像颜色空间转换为HSV格式,
然后计算亮度和饱和度的均值，
具体见demo中的hsv.py。
"""

# 装饰器：记录图像属性
def ImageHSV(log_file='image_alert.log'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数获取图像
            img = func(*args, **kwargs)
            
            if img is not None:
                # 计算图像属性
                size = img.shape[:3]  # (height, width, channels)
                
                # 转换到HSV颜色空间
                hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                
                # 计算亮度和饱和度
                brightness = np.mean(hsv_image[:, :, 2])
                saturation = np.mean(hsv_image[:, :, 1])
                
                # 记录日志
                logging.basicConfig(
                    filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s'
                )
                logging.info(
                    f"Image captured - Size: {size}, "
                    f"Brightness: {brightness:.2f}, "
                    f"Saturation: {saturation:.2f}"
                )
            
            return img
        return wrapper
    return decorator

"""
2. 实现一个函数装饰器img_resizer,
其能够对返回读取并返回图片的其他函数进行装饰,
目的是将所返回的图片压缩到特定的大小。
比如在利用大模型进行图片内容理解时，
为了节约成本,有时需要在不影响理解准确率的前提下,尽量减少输入的token数。
"""

# 装饰器：调整图像大小
def img_resizer(target_size=(640, 480), keep_aspect_ratio=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            img = func(*args, **kwargs)
            
            if img is not None:
                h, w = img.shape[:2]
                
                if keep_aspect_ratio:
                    target_w, target_h = target_size
                    scale = min(target_w / w, target_h / h)
                    new_size = (int(w * scale), int(h * scale))
                else:
                    new_size = target_size
                
                img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
            
            return img
        return wrapper
    return decorator
    
"""
参照ark_image_cap.py,
实现一个基于大模型的图像预警类ImageAlter,
其能够定时地从摄像头采集照片，
并提交给大模型理解语义，
当某种特殊情形发生时
（如有人出现、有刀等出现、有交通工具出现、有猫、狗等动物出现、有特定行为
（睡觉、化妆、打字、扫地等出现），
通过某种形式进行输出，
并保留包含特殊情形的照片。注意，
在实现该类时，
要对1-3中的各个装饰器进行恰当的使用和测试;
场景自定义，测试要真实，可以选择室内或室外，体现实用性和趣味性。
特别地,要注意采集频率跟token的关系,体会成本与实用性之间的折中。
"""

class ImageAlert:
    def __init__(self, api_key, model_id, alert_interval=10, save_dir="alerts"):
        """
        初始化图像预警系统
        
        参数:
            api_key: 火山引擎API密钥
            model_id: 大模型ID
            alert_interval: 预警检测间隔(秒)
            save_dir: 预警图片保存目录
        """
        self.client = Ark(api_key=api_key)
        self.model_id = model_id
        self.alert_interval = alert_interval
        self.save_dir = save_dir
        self.last_alert_time = 0
        self.alert_items = [
            "人", "刀", "武器", "交通工具", "汽车", "自行车", 
            "猫", "狗", "动物", "睡觉", "化妆", "打字", "扫地"
        ]
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化摄像头
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

    @ImageHSV()
    @img_resizer(target_size=(640, 480))
    def capture_image(self):
        """捕获当前帧图像"""
        ret, frame = self.cap.read()
        return frame if ret else None

    def encode_image(self, img):
        """将OpenCV图像转为Base64编码"""
        _, buffer = cv.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_image(self, img):
        """使用大模型分析图像内容"""
        base64_image = self.encode_image(img)
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请详细描述图片内容，包括物体、人物、动物及其行为。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def check_alert(self, description):
        """检查描述中是否包含预警项"""
        for item in self.alert_items:
            if item in description:
                return True, item
        return False, None

    def save_alert_image(self, img, alert_item):
        """保存触发预警的图像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/alert_{alert_item}_{timestamp}.jpg"
        cv.imwrite(filename, img)
        print(f"预警图像已保存: {filename}")

    def run(self):
        """运行预警系统"""
        print("图像预警系统启动...")
        print(f"监测项: {', '.join(self.alert_items)}")
        
        try:
            while True:
                # 捕获图像
                img = self.capture_image()
                if img is None:
                    print("无法获取图像")
                    time.sleep(1)
                    continue
                
                # 显示实时画面
                cv.imshow('Live Feed', img)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 控制检测频率
                current_time = time.time()
                if current_time - self.last_alert_time < self.alert_interval:
                    time.sleep(0.1)
                    continue
                
                # 分析图像
                try:
                    description = self.analyze_image(img)
                    print(f"分析结果: {description}")
                    
                    # 检查预警
                    alert, alert_item = self.check_alert(description)
                    if alert:
                        print(f"⚠️ 预警触发: 检测到 {alert_item}")
                        self.save_alert_image(img, alert_item)
                        self.last_alert_time = current_time
                
                except Exception as e:
                    print(f"分析失败: {str(e)}")
                
        finally:
            self.cap.release()
            cv.destroyAllWindows()
            print("系统已关闭")



# 使用示例
if __name__ == "__main__":
    # 配置参数
    API_KEY = '570d6def-96e8-465d-ab2c-55b38d7773c9'
    MODEL_ID = "doubao-1-5-vision-pro-32k-250115"
    
    # 创建并运行预警系统
    alert_system = ImageAlert(
        api_key=API_KEY,
        model_id=MODEL_ID,
        alert_interval=15,  # 15秒检测一次以控制成本
        save_dir="alert_images"
    )
    alert_system.run()