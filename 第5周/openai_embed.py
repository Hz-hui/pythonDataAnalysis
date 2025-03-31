#text-embedding-3-small  62,500  62.3%   8191
#text-embedding-3-large  9,615   64.6%   8191
#text-embedding-ada-002  12,500  61.0%   8191

key = '不能给别人看'

#pip install openai
from openai import OpenAI
client = OpenAI(api_key = key)

#从下面加循环或者并行，不一定需要并行，如果速度还可以的话。

response = client.embeddings.create(
    input="2013年是公司成立以来经济效益最好、增长速度最快的一年，公司董事会认真履行职责，及时召集召开董事会及股东 大会会议，决策公司重大事项，完善公司治理体系，督促管理层努力夯实安全生产基础，不断加快项目开发建设，持续提升 精细化管理水平，公司建设和经营取得显著成果。",
    model="text-embedding-3-large"
)
print(len(response.data[0].embedding))#3072维
print(response.data[0].embedding)