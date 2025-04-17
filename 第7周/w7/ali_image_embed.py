#阿里百炼 pip install dashscope
import dashscope
import base64
from ali_key import KEY, MODEL #注意，这个模块要自己构建，里面有自己的KEY和选择的模型名称。

import json
from http import HTTPStatus

import sys

# 读取图片并转换为Base64
image_path = sys.argv[1]
with open(image_path, "rb") as image_file:
    # 读取文件并转换为Base64
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

image_format = sys.argv[2]  # 根据实际情况修改，比如png, jpg、bmp 等
image_data = f"data:image/{image_format};base64,{base64_image}"

# 输入数据
inputs = [{'image': image_data}]

# 调用模型接口
resp = dashscope.MultiModalEmbedding.call(
    api_key = KEY,
    model = MODEL,
    input = inputs
)
if resp.status_code == HTTPStatus.OK:
    print(resp.output['embeddings'][0]['embedding'])#1024维向量
    #print(json.dumps(resp.output, ensure_ascii=False, indent=4))