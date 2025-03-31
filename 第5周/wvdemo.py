import jieba
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np#开始接触这个库，后面也会专门再讲一次

#windows在C盘的字体文件夹里
STH = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

# 定义语料库（省略文本预处理）
stopset = set(['[',']','@'])
sentences = []
with open('weibo_demo.txt','r') as f:
    for line in f:
        sen = [w for w in jieba.cut(line.strip().split('\t')[1]) if w not in stopset]
        #print(sen)
        sentences.append(sen)
print(f'load {len(sentences)} tweets...')


# 建立Word2Vec模型
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1)
# 模型的保存
# model.save("word2vec.model")

# 预训练模型的加载
#model = Word2Vec.load("weibo_59g_embedding_200.model")
#print(model.wv.most_similar("春天", topn=5))

# 获取最相关的10个词和最不相关的10个词
most_similar = model.wv.most_similar("花朵", topn=10)
least_similar = model.wv.most_similar(negative=["花朵"], topn=10)

print(most_similar)
print(least_similar)

# 将最相关和最不相关的词汇向量合并为一个数组
vectors = np.array([model.wv[word] for word, similarity in most_similar + least_similar])
print(vectors.shape)
words = [word for word, similarity in most_similar + least_similar]

# 使用t-SNE算法对词向量进行降维
tsne = TSNE(n_components=2, perplexity=15)
#print(vectors)
vectors_tsne = tsne.fit_transform(vectors)

# 可视化降维后的词向量
fig, ax = plt.subplots()
ax.set_title('花朵', fontproperties = STH)
# t-SNE降维至2维
ax.scatter(vectors_tsne[:10, 0], vectors_tsne[:10, 1], color='blue',label = 'most_simi')
ax.scatter(vectors_tsne[10:, 0], vectors_tsne[10:, 1], color='red', label = 'least_simi')
ax.legend()
# 打印词
for i, word in enumerate(words):
    ax.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]),fontproperties = STH)
plt.show()
