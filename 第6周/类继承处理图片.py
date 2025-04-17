'''
在进行一些图片数据分析任务，我们常常需要对图片进行预处理，
如将批量的图片处理为相似的格式、尺寸，或者从图片中抽取文本以便后续其他分析。
因此，本次作业要求基于类的继承，初步了解和掌握python对图像的简单处理；
并通过大模型API，实现图片文本的抽取尝试。
'''
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from volcenginesdkarkruntime import Ark
from ark_key_model import KEY as API_KEY
from ark_key_model import MODEL as ARK_MODEL
import base64
import os
import sys

'''
1. 实现基类ImageProcessor。

（1）至少包括两个数据属性：待处理的图片实例（即PIL库的Image实例）以及一个参数列表（用于存储图片处理时需要的参数）；

（2）至少包括一个方法：process()，用于对Image实例进行特定处理。这一方法将会在子类中进行实现，不需要在该类中实现。
'''


class ImageProcessor:
    def __init__(self, image, params=None):
        """
        初始化ImageProcessor实例。

        :param image: PIL库的Image实例，表示待处理的图片。
        :param params: 参数列表，用于存储图片处理时需要的参数。默认为None。
        """
        if params is None:
            params = []
        
        self.image = image
        self.params = params

    def process(self):
        pass


'''
2. 实现四个ImageProcessor的子类，类名自定义，体现继承关系。

（1）分别完成对于图片的灰度化处理、裁剪或调整大小、模糊以及边缘提取四类操作；

（2）分别对process()方法进行实现（实际上只需要简单的调用用PIL中的Image和ImageFilter方法等即可实现，具体见本周demo中的imp.py）。
'''
class GrayscaleProcessor(ImageProcessor):
    def process(self):
        """将图片转换为灰度图像。"""
        return self.image.convert('L')
 
class CropResizeProcessor(ImageProcessor):
    def process(self):
        """根据参数裁剪或调整图片大小。
        假设参数是一个元组，格式为 (action, size_or_box)，
        其中 action 是 'crop' 或 'resize'，size_or_box 是裁剪框或新尺寸。
        """
        action, size_or_box = self.params
        if action == 'crop':
            return self.image.crop(size_or_box)
        elif action == 'resize':
            return self.image.resize(size_or_box)
        else:
            raise ValueError("无效的操作: {}".format(action))
 
class BlurProcessor(ImageProcessor):
    def process(self):
        """对图片进行模糊处理。"""
        return self.image.filter(ImageFilter.BLUR)
 
class EdgeProcessor(ImageProcessor):
    def process(self):
        """对图片进行边缘提取处理。"""
        return self.image.filter(ImageFilter.CONTOUR)  # 或者使用 EDGE_ENHANCE_MORE
''' 
# 示例用法
if __name__ == "__main__":
    # 打开一张图片
    im = Image.open('第6周\w6\lx.jpg')
    print(im.format, im.size, im.mode)  # size width, height
 
    # 灰度化处理
    gray_processor = GrayscaleProcessor(im)
    gray_image = gray_processor.process()
 
    # 裁剪或调整大小（这里选择裁剪作为示例）
    crop_box = (10, 10, 250, 250)
    crop_resize_processor = CropResizeProcessor(im, params=('crop', crop_box))
    cropped_image = crop_resize_processor.process()
 
    # 模糊处理
    blur_processor = BlurProcessor(im)
    blurred_image = blur_processor.process()
 
    # 边缘提取处理
    edge_processor = EdgeProcessor(im)
    edge_image = edge_processor.process()
 
    # 显示处理后的图片
    plt.figure()
 
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale')
 
    plt.subplot(2, 2, 2)
    plt.imshow(cropped_image)
    plt.title('Cropped')
 
    plt.subplot(2, 2, 3)
    plt.imshow(blurred_image)
    plt.title('Blurred')
 
    plt.subplot(2, 2, 4)
    plt.imshow(edge_image)
    plt.title('Edge')
 
    plt.show()
 
    # 保存边缘提取后的图片
    edge_image.save('lx_contour.png')

'''
'''
3. 实现ImageProcessor的子类 ImageTextExtrator
，利用大模型API实现图片文字的抽取。
(具体见本周demo中的ark_dmeo.py，注意，需要选择一个大模型并创建api_key，
可以用豆包等，有一定规模的免费token，具体见https://www.volcengine.com/docs/82379/1362931。)
'''
class ImageTextExtractor(ImageProcessor):
    def __init__(self, api_key):
        super().__init__()  # 调用父类的构造函数，如果有必要
        self.client = Ark(api_key=api_key)
 
    def encode_image(self, image_path):
        """将指定图片转为Base64编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
 
    def extract_text_from_image(self, image_path, image_format='jpeg'):
        """从图片中提取文字"""
        base64_image = self.encode_image(image_path)
        
        # 构造图片URL，根据图片格式添加前缀
        image_url_prefix = f"data:image/{image_format};base64,"
        image_url = f"{image_url_prefix}{base64_image}"
        
        response = self.client.chat.completions.create(
            model=ARK_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "识别图片中的文字。",  # 通过文本prompt明确图片分析的任务
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
        )
        
        return response.choices[0].message.content
 
# 使用示例
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]  # 从命令行参数获取图像路径
    extractor = ImageTextExtractor(api_key=API_KEY)
    extracted_text = extractor.extract_text_from_image(image_path, image_format='jpeg')  # 根据实际情况修改图片格式
    print(extracted_text)