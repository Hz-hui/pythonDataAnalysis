#这个模块不作为DEMO，：）
from dskey import DEEPSEEK_GEN_KEY as KEY

from openai import OpenAI
import sys

client = OpenAI(api_key=KEY, 
    base_url="https://api.deepseek.com")

#通过指定 model='deepseek-chat' 即可调用 DeepSeek-V3。
#通过指定 model='deepseek-reasoner'，即可调用 DeepSeek-R1。
EMOTION_LABEL={'0':'消极',
                '1':'积极',
                '2':'中性'}
#注意，这里仅示意，没有太在意成本的计算，一次传一条。
#如果批量处理，建议用批量相关的api，或者一次给多条数据。
#一般max_token的数目比较大，可以给多条。
DATA_PROMPT = '''现在，我给你一条微博文本，\\
请对其进行基本的文本预处理后，返回其表达的情绪，是积极、消极还是中性。如果是积极，请返回1，\\
如果是消极，请返回0；如果是中性，返回2。注意，只需要返回1、0或者2。'''
index = 0 #只示例前10条
txts=[]
with open('weibo_demo.txt','r') as rf:
    for line in rf:
        if index >=10: break
        txt = line.strip().split('\t')[1]
        txts.append(txt)
        index += 1
print(len(txts))
for txt in txts:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '你是一个数据分析工程师。'},
            {"role": "user", "content": DATA_PROMPT+'\n'+txt},
        ],
        
        #这个参数特别要注意，建议参阅官方文档，deepseek的值特别“大”，
        #官方说明：https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings
        
        temperature=1.0,
        stream=False
        )
    r = response.choices[0].message.content
    print(f'{txt}\t{EMOTION_LABEL[r]}')