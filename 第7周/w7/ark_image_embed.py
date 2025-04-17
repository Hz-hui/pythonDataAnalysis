import requests
import json
import sys
import base64
from ark_key_model import KEY

# 接口地址
url = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}"
}

image_path = sys.argv[1]
with open(image_path, "rb") as image_file:
    # 读取文件并转换为Base64
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

image_format = sys.argv[2]  # 根据实际情况修改，比如png, jpg、bmp 等

# 请求体
# 注意要先去火山大模型的控制台，开通对应的模型，这个操作特别麻烦，不知道为什么这么设计，别的公司没有这一步
# 从模型广场出发，先找到doubao-embedding-vision，完了去开通即可
payload = {
    "model": "doubao-embedding-vision-241215", #vision的嵌入目前只有这一个模型
    "input": [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_format};base64,{base64_image}"
            }
        }
    ]
}

# 发送 POST 请求，注意payload也是以json的格式发送到大模型服务器
response = requests.post(url, headers=headers, data=json.dumps(payload))

#response.json()里还有其他信息，可直接打印来查看
#print(response.json())
embedding = response.json()['data']['embedding']

print(len(embedding))#3072维

# 打印响应结果
#print("Status Code:", response.status_code)
print("Embedding:", embedding)
