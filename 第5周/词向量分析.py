import jieba
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np  #开始接触这个库，后面也会专门再讲一次
import jieba.posseg as pseg
from collections import Counter
import os
from matplotlib import rcParams

# 设置字体路径（假设使用 SimHei 字体）
font_path = 'C:\\Windows\\Fonts\\simhei.ttf'  # Windows 下的 SimHei 字体路径
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

stopwords_path = 'C:\\Users\\13261\\Desktop\\大二\\大二下课程内容\\Python\\pythonDataAnalysis\\第2周\\cn_stopwords.txt'

'''
Word2vec是一类用来产生词向量的神经网络模型，
通过学习文本语料库中单词之间的语义和上下文关系来表示单词的向量空间。
在训练完成后，它可以把每个词映射到一个可以表示词与词之间关系的向量。
因此它可以被用来进行文本聚类和相似度计算等任务，也常常被应用在包括
文本分类、情感分析、信息检索、机器翻译等场景下。
因此，参照wvdemo.py，本次作业将实现一个简单的类，
并体会word2vec的各种相关应用。
'''


'''
1. 定义一个类TextAnalyzer，
其属性包括待分析的文本文件路径，
等加载的预训练模型文件路径，
训练word2vec的一些简单参数
（如向量长度，窗口大小）等，
初始化的时候需要对这些属性进行定义。
'''


class TextAnalyzer():
    def __init__(self, file_path, stopwords_path, modal_path, vec_length, window_size):
        self.file_path = file_path
        self.modal_path = modal_path
        self.vec_length = vec_length
        self.window_size = window_size
        self.stopwords_path = stopwords_path
        self.stop_words = set()
        if stopwords_path:
            self._load_stopwords()

    '''
    2. 在上述类加入一个预处理方法_pre_process，
    如将待分析的weibo.txt加载到内存（请先解压提供的weibo.txt.zip)，
    进行基本的文本预处理，如对所有微博内容进行去重，进行分词、去除停用词、标点等，
    最终建立一个以微博为单位进行分词的二维列表。
    注意，weibo.txt一行为一条微博的属性，用\t分隔后，第二个元素为微博内容。
    （提供的weibo.txt包含大量重复和标点等，需要仔细预处理，否则会影响后面的嵌入模型训练。）
    '''
    def _load_stopwords(self):
        with open(self.stopwords_path, 'r', encoding='UTF-8') as f:
            for line in f:
                self.stop_words.add(line.strip())
 
    def _pre_process(self):
        segmented_lines = []  # 用于存储每行的分词结果
        line_word_freqs = []  # 用于存储每行的词频统计
 
        # 打开文件并逐行读取
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 移除行末的换行符并按制表符分割
                parts = line.strip().split('\t')
                
                # 检查该行是否有至少两个元素
                if len(parts) > 1:
                    content = parts[1]  # 微博内容
 
                    # 分词
                    terms = jieba.cut(content)
                    segmented_line = list(terms)  # 将分词结果转换为列表
                    segmented_lines.append(segmented_line)
 
                    # 过滤停用词并统计词频
                    filtered_terms = [word for word in segmented_line if word not in self.stop_words and word not in {'\n', ' '}]
                    word_freq = Counter(filtered_terms)
                    line_word_freqs.append(word_freq)
 
        '''
        # 打印每行的词频统计
        for i, word_freq in enumerate(line_word_freqs):
            print(f"Line {i+1} word frequencies:")
            for word, freq in word_freq.items():
                print(f"{word}\t{freq}")
            print()
        '''
 
        return segmented_lines
    
    
    '''
    3. 在上述类加入一个方法
    _get_word2vec_model来利用2中构建的微博二维列表来训练Word2Vec模型，
    可参照demo使用gensim完成Word2Vec模型的建立。
    '''

    def _get_word2vec_model(self, sentences):
        model = Word2Vec(sentences=sentences, vector_size=self.vec_length, window=self.window_size, min_count=10, workers=4)
        return model

    '''
    4. 在上述类加入一个方法get_similar_words来利用训练得到的word2vec模型来推断相似词汇。
    如输入一个任意词，使用model.similarity方法可以来返回与目标词汇相近的一定数目的词。
    '''

    def get_similar_words(self, model, word, topn=10):
        try:
            similar_words = model.wv.most_similar(word, topn=topn)
            return similar_words
        except KeyError:
            return f"The word '{word}' is not in the vocabulary."

    '''
    7. 词向量能够将每个词定位为高维空间中的一个点，
    且不同词间的“差异”可以通过点间的距离来反应。
    在类加再增加一个方法vis_word_tsne，
    使用Demo中演示的TSNE算法，
    对与某个输入词的最相关以及最不相关的词语（用参数来控制数目）
    进行降维和可视化。
    '''    
    def get_dissimilar_words(self, model, word, topn=10):
        # 简单实现：选择距离最远的词（通过计算余弦相似度取负值）
        try:
            all_words = list(model.wv.index_to_key)
            word_vector = model.wv[word]
            distances = []
            for w in all_words:
                if w != word:
                    similarity = model.wv.similarity(word, w)
                    distance = 1 - similarity  # 简单地使用 1 减去相似度作为距离
                    distances.append((w, distance))
            distances.sort(key=lambda x: x[1], reverse=True)
            dissimilar_words = distances[:topn]
            return dissimilar_words
        except KeyError:
            return f"The word '{word}' is not in the vocabulary."

    def vis_word_tsne(self, model, word, similar_topn=5, dissimilar_topn=5):
        similar_words = self.get_similar_words(model, word, topn=similar_topn)
        dissimilar_words = self.get_dissimilar_words(model, word, topn=dissimilar_topn)
    
        print(f"Most similar to '{word}':", similar_words)
        print(f"Most dissimilar to '{word}':", dissimilar_words)

        if isinstance(similar_words, str) or isinstance(dissimilar_words, str):
            print(similar_words if isinstance(similar_words, str) else dissimilar_words)
            return
 
        # 合并相似词和不相似词
        words_to_plot = similar_words + dissimilar_words
        words = [word_sim[0] for word_sim in words_to_plot]
        vectors = np.array([model.wv[word_sim[0]] for word_sim in words_to_plot])  # 转换为 NumPy 数组
 
        # 标记相似词和不相似词
        labels = ['相似' if i < similar_topn else '不相似' for i in range(len(words))]
 
        # t-SNE 降维
        perplexity = min(15, len(vectors) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        vectors_tsne = tsne.fit_transform(vectors)
 
        # 可视化
        fig, ax = plt.subplots()
        ax.set_title(f'Word Embeddings for "{word}"')
 
        # 使用不同颜色区分相似词和不相似词
        colors = ['blue' if label == '相似' else 'red' for label in labels]
        ax.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], c=colors)
 
        for i, word_sim in enumerate(words):
            ax.annotate(word_sim, (vectors_tsne[i, 0], vectors_tsne[i, 1]))
 
        plt.show()


weibo = TextAnalyzer('第5周\weibo_demo.txt', stopwords_path, '第5周', vec_length=50, window_size=5)
segmented_lines = weibo._pre_process()

# 路径设置（确保路径正确）
file_path = os.path.join('C:\\', 'Users', '13261', 'Desktop', '大二', '大二下课程内容', 'Python', 'pythonDataAnalysis', '第5周', 'weibo_demo.txt')
stopwords_path = os.path.join('C:\\', 'Users', '13261', 'Desktop', '大二', '大二下课程内容', 'Python', 'pythonDataAnalysis', '第2周', 'cn_stopwords.txt')
 
# 打印预处理后的分词结果数量
print(f'Loaded {len(segmented_lines)} lines of segmented text.')
 
# 建立 Word2Vec 模型
model = weibo._get_word2vec_model(segmented_lines)

# 输入要查询的词
word = input("请输入要查询的词：")  # eg：花朵
 
# 可视化最相关和最不相关的词
weibo.vis_word_tsne(model, word, similar_topn=20, dissimilar_topn=20)
