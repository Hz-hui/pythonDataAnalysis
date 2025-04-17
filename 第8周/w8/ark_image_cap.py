import base64
import os
import sys
from volcenginesdkarkruntime import Ark

from ark_key_model import KEY as API_KEY
from ark_key_model import MODEL as ARK_MODEL

# 初始化一个Client对象，提供API Key
client = Ark(api_key = API_KEY)

# 定义方法将指定图片转为Base64编码
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# 需要传给大模型的图片
image_path = sys.argv[1]
image_format = sys.argv[2]#格式,jpg等

# 将图片转为Base64编码
base64_image = encode_image(image_path)

response = client.chat.completions.create(
  # 模型的Model ID
  model = ARK_MODEL,
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "请详细地描述图片内容。"
        },
        {
          "type": "image_url",
          "image_url": {
          "url":  f"data:image/{image_format};base64,{base64_image}"
          },
        },
      ],
    }
  ],
)
print(response.choices[0].message.content)