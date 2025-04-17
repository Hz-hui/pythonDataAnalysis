from PIL import Image
import PIL
import numpy as np
import scipy.stats as stats
import dashscope
import base64
import json
from http import HTTPStatus
from sklearn.metrics.pairwise import cosine_similarity
import os
KEY = "sk-4a978445232c4332976ccbc85fc97647"


class imageQueryError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
class ImageQueryShapeNotMatchError(imageQueryError):
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2
        self.message = f"图像形状不匹配: {shape1} vs {shape2}"
        super().__init__(self.message)

class ImageQuery():
    def __init__(self, image_path):
        self.image_path = image_path
        try:    
            self.image = Image.open(image_path)
        except FileNotFoundError:
            print(f"文件未找到: {image_path}")
            self.image = None
        except PIL.UnidentifiedImageError:
            print(f"无法识别的图像文件: {image_path}")
            self.image = None
        except Exception as e:
            print(f"发生错误: {e}")
            self.image = None
        else:
            print(f"成功打开图像: {image_path}")
            
    def pixel_difference(example_1,example_2):
        image1 = example_1.image
        image2 = example_2.image
        array1 = np.array(image1)
        array2 = np.array(image2)
        if array1.shape != array2.shape:
            raise ImageQueryShapeNotMatchError(array1.shape, array2.shape)
        diff = np.abs(array1 - array2)
        # 计算绝对差异的均值
        mean_diff = np.mean(diff)
        return mean_diff
    def get_histogram(self):
        if self.image is None:
            print("无法获取直方图：图像无效")
            return None
        histogram = self.image.histogram()
        return histogram

    def pearson_correlation(self, other_image):
        if self.image is None or other_image.image is None:
            print("无法比较：至少有一个图像无效")
            return None
        hist1 = self.get_histogram()
        hist2 = other_image.get_histogram()
        if hist1 is None or hist2 is None:
            print("无法计算相关性：直方图无效")
            return None
        correlation, p_value = stats.pearsonr(hist1, hist2)
        return correlation, p_value
    
    def spearman_correlation(self, other_image):
        if self.image is None or other_image.image is None:
            print("无法比较：至少有一个图像无效")
            return None
        hist1 = self.get_histogram()
        hist2 = other_image.get_histogram()
        if hist1 is None or hist2 is None:
            print("无法计算相关性：直方图无效")
            return None
        correlation, p_value = stats.spearmanr(hist1, hist2)
        return correlation, p_value

    def kendall_correlation(self, other_image):
        if self.image is None or other_image.image is None:
            print("无法比较：至少有一个图像无效")
            return None
        hist1 = self.get_histogram()
        hist2 = other_image.get_histogram()
        if hist1 is None or hist2 is None:
            print("无法计算相关性：直方图无效")
            return None
        correlation, p_value = stats.kendalltau(hist1, hist2)
        return correlation, p_value
    

    def ali_embedding(self):
        if self.image is None:
            print("无法获取嵌入：图像无效")
            return None
        with open(self.image_path, "rb") as image_file:
            # 读取文件并转换为Base64
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        # 设置图像格式
        image_format = "jpg"  # 根据实际情况修改，比如jpg、bmp 等
        image_data = f"data:image/{image_format};base64,{base64_image}"
        # 输入数据
        inputs = [{'image': image_data}]
        resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        api_key=KEY,
        input=inputs
        )
        if resp.status_code == HTTPStatus.OK:
            self.embedding_vector = resp.output['embeddings'][0]['embedding']
            return resp.output['embeddings'][0]['embedding']
        

        
    def embed_correlation(self, other_image):
        if self.image is None or other_image.image is None:
            print("无法比较：至少有一个图像无效")
            return None
        embedding1 = self.ali_embedding()
        embedding2 = other_image.ali_embedding()
        if embedding1 is None or embedding2 is None:
            print("无法计算相关性：嵌入无效")
            return None
        array1 = np.array(embedding1).reshape(1, -1)
        array2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(array1, array2)[0][0]
        return similarity
    
    def average_hash(self):
        if self.image is None:
            print("无法计算平均哈希：图像无效")
            return None
        return imagehash.average_hash(self.image)
    
    def phash(self):
        if self.image is None:
            print("无法计算感知哈希：图像无效")
            return None
        return imagehash.phash(self.image)
    def dhash(self):
        if self.image is None:
            print("无法计算差异哈希：图像无效")
            return None
        return imagehash.dhash(self.image)
    def whash(self):
        if self.image is None:
            print("无法计算加权哈希：图像无效")
            return None
        return imagehash.whash(self.image)
    
    def _creat_an_image(file_path):
        try:
            image = Image.open(file_path)
            return image
        except Exception as e:
            print(f"无法创建图像: {file_path}, 错误: {e}")
            return None
        
    def load_images(images):
        image_list = {}
        for image_path in images:
            filename = os.path.basename(image_path)
            image= ImageQuery._creat_an_image(image_path)
            if image is not None:
                image_list[filename] = image
            else:
                print(f"无法加载图像: {image_path}")
        return image_list
        
image1= ImageQuery('第7周\wx1.png')
image2= ImageQuery('第7周\wx2.png')
print("像素绝对值差平均值：")
print(ImageQuery.pixel_difference(image1,image2))
print("pearson相关性:")
print(image1.pearson_correlation(image2))
print("spearman相关性:")
print(image1.spearman_correlation(image2))
print("kendall相关性:")
print(image1.kendall_correlation(image2))

'''
print("embed_correlation:")
print(image1.embed_correlation(image2))

hash1 = image1.average_hash()
hash2 = image2.average_hash()
distance = hash1 - hash2
print(f"平均哈希距离: {distance}")

list1 = [r"ceshi\example_1.jpg",r"ceshi\example_2.jpg",r"ceshi\example_3.jpg",r"ceshi\example_4.jpg",r"ceshi\example_5.jpg"]
imagedic=ImageQuery.load_images(list1)
for name,image in imagedic.items():
    print(f"{name}: {image}")

'''