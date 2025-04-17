'''
PIL能够实现许多图片数据的底层表示和处理，
比如在相素层面进行相关分析等，
并支撑常用的图像检索任务。
请围绕其相关功能，
结合异常捕获和自定义异常，
完成如下题目。
'''
import PIL
from PIL import Image
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import dashscope
import base64
import json
from http import HTTPStatus
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from ali_key import KEY

class ImageQueryError(Exception):
	def __init__(self):
		pass

class ImageQueryShapeNotMatchError(ImageQueryError):
    def __init__(self,image_path1,image_path2):

        # 打开第一张图片
        img1 = Image.open(image_path1)
        # 获取第一张图片的尺寸
        size1 = img1.size

        # 打开第二张图片
        img2 = Image.open(image_path2)
        # 获取第二张图片的尺寸
        size2 = img2.size

        self.message= f"大小不对，img1的大小是{size1}, img2的大小是{size2}"


class ImageQuery:
    '''
    1. 异常捕获。
    实现ImageQuery类的_create_and_image方法，
    其利用PIL.Image类的open方法打开并返回一个Image实例，
    但考虑到open方法可能产生FileNotFoundError或PIL.UnidentifiedImageError，
    请在该方法中对这两个异常进行捕获和处理（打印或记入日志，相关信息包括打开的文件路径和详细的异常描述）。
    '''
    def __init__(self, image_path=None):
        self.image = self._create_and_image(image_path) if image_path else None
        self.histogram = []

    def _create_and_image(self, fp):
        try:
            return Image.open(fp)
        except FileNotFoundError as e:
            print(f"文件未找到错误: {e}")
            return None
        except PIL.UnidentifiedImageError as e:
            print(f"无法识别的图片格式: {e}")
            return None

    '''
    2. 图片的相似性计算。
    在ImageQuery类中实现一种简单图片相似性的计算方法pixel_difference，
    即直接对两个图片逐相素相减，并累积求和差异的绝对值，
    继而除以相素总数。
    注意该方法可能会抛出一个叫ImageQueryShapeNotMatchError的自定义异常，
    其继承了ImageQueryError（本次作业自定义的顶层异常类），
    即当比较相似性的两张图片形状（长宽）不一致性时。请在该方法中抛出该异常，包含两个图片的形状信息。
    '''   
    def pixel_difference(image_path1, image_path2):
        # 打开两张图片
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)
        
        # 确保两张图片的尺寸相同
        if img1.size != img2.size:
            raise ImageQueryShapeNotMatchError(image_path1, image_path2)
        
        else:
            # 将图片转换为NumPy数组
            img1_array = np.array(img1, dtype=np.float32)
            img2_array = np.array(img2, dtype=np.float32)
            
            # 计算像素差异绝对值
            difference = np.abs(img1_array - img2_array)
            
            # 计算差异的总和
            total_difference = np.sum(difference)
            
            # 计算像素总数
            total_pixels = img1_array.size  # 直接使用展平后的元素数量
            
            # 计算平均差异
            average_difference = total_difference / total_pixels
            
            return average_difference
        
        
    '''
    3. 图片的直方图相似性计算。
    在ImageQuery类中实现更多的相似性计算方法。
    具体地，利用PIL.Image类的histogram方法，
    获取图片相素的直方图，
    进而用scipy.states中的相关性计算方法来得到不同的相似性，
    如pearson，spearman，kendall等。
    这些方法并不要求图片形状一致。注意，这些相似性方法还能够返回显著性。
    '''
    
    def get_histogram(self):
        histogram = self.image.histogram()
        self.histogram = histogram
        return histogram

    
    def compute_similarity(self, other_image_query, method='pearson'):
        """
        计算与另一个ImageQuery对象的相似性。
        
        :param other_image_query: 另一个ImageQuery对象
        :param method: 相似性计算方法，可选 'pearson', 'spearman', 'kendall'
        :return: 相似性系数和显著性水平（p值）
        """

        if method == 'pearson':
            correlation, p_value = pearsonr(ImageQuery.get_histogram(self), ImageQuery.get_histogram(other_image_query))
        elif method == 'spearman':
            correlation, p_value = spearmanr(ImageQuery.get_histogram(self), ImageQuery.get_histogram(other_image_query))
        elif method == 'kendall':
            correlation, p_value = kendalltau(ImageQuery.get_histogram(self), ImageQuery.get_histogram(other_image_query))
        else:
            raise ValueError("Unsupported method. Choose from 'pearson', 'spearman', 'kendall'.")
        
        return correlation, p_value
    
    '''
    4. 图片的大模型嵌入。
    在ImageQuery类中实现基于大模型的相似性计算方法，
    即利用相关API(具体见Demo ali_image_embed.py或者ark_image_embed.py)
    首先将图片嵌入为向量，继而通过向量的余弦相似度等给出相似性大小(cos_simi.py)。
    注意，选一个大模型实现即可，ali和字节均提供一定的免费token额度。
    '''
    
    def read_image_as_base64(self):
        if self.image is None:
            return None
        from io import BytesIO
        buffered = BytesIO()
        self.image.save(buffered, format=self.image.format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


image_path1 = '第7周/w7/test.png'
image_path2 = '第7周/w7/test2.jpg'
image_path3 = '第7周/w7/test3.jpeg'
image_path4 = '第7周/wx1.png'
image_path5 = '第7周/wx2.png'

try:
    image_query1 = ImageQuery(image_path1)
    image_query2 = ImageQuery(image_path2)
 
    # 计算像素差异
    diff = ImageQuery.pixel_difference(image_path4, image_path5)
    print(f"Pixel difference: {diff}")
 
    # 计算直方图相似性
    pearson_similarity, pearson_p_value = image_query1.compute_similarity(image_query2)
    print(f"Pearson similarity: {pearson_similarity}, p-value: {pearson_p_value}")

    
    # 读取图片并转换为Base64
    base64_image_1 = image_query1.read_image_as_base64()
    image_data1 = f"data:image/{image_path1};base64,{base64_image_1}"
    base64_image_2 = image_query2.read_image_as_base64()
    image_data2 = f"data:image/{image_path2};base64,{base64_image_2}"
 
    # 输入数据
    inputs1 = [{'image': image_data1}]
    inputs2 = [{'image': image_data2}]

    # 调用模型接口
    resp1 = dashscope.MultiModalEmbedding.call(
        api_key = KEY,
        model="multimodal-embedding-v1",
        input = inputs1
    )
    if resp1.status_code != HTTPStatus.OK:
        raise Exception(f"API call 1 failed with status code {resp1.status_code}")
    a = resp1.output['embeddings'][0]['embedding']

    resp2 = dashscope.MultiModalEmbedding.call(
        api_key = KEY,
        model="multimodal-embedding-v1",
        input = inputs2
    )
    if resp2.status_code != HTTPStatus.OK:
        raise Exception(f"API call 2 failed with status code {resp2.status_code}")
    b = resp2.output['embeddings'][0]['embedding']

    # 计算余弦相似度
    similarity = cosine_similarity([a], [b])  # 注意这里需要将向量包装为2D数组

    # 输出是一个矩阵,如果只比较两个向量，结果是 1x1 矩阵，所以取 [0][0]。
    print("余弦相似度(test.jpg, test2.jpg):", similarity[0][0])

except ImageQueryShapeNotMatchError as iqsnme:
    print(iqsnme.message)
except Exception as e:
    print(f"An error occurred: {e}")