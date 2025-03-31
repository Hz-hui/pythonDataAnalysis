'''
LDA（Latent Dirichlet allocation）是一种常用的主题模型，
它可以将文档集中每篇文档的主题按照概率分布的形式给出。它是一种无监督学习算法，
在训练时仅需要输入文档集并给定主题数量。
这一模型目前在文本挖掘，包括文本主题识别、文本分类以及文本相似度计算方面均有应用。
请利用week.csv提供的广州八大热门糖水店的评论数据，
进一步对评论文本（即cus_comment）进行话题识别与分析。
注意，本周使用了两类方法来实现lda(sklearn和gensim)，本次作业选自己喜欢的来实现即可。

1. 文档预处理。

一般来讲，LDA在评论等短文本上的效果并不理想，
且多数情况下，我们希望给话题赋予时间含义，以讨论其“波动性”。
因此，往往先需要按时间进行文档的生成，
比如，将某一店铺的评论按年进行合并，即将某店铺某年发布的所有评论视为一个文档。
请实现一个模块，其中包含一个或多个函数，其能够读取该数据集并将之分店铺
（共8家店铺，可根据shopID进行区分）处理以天（或其他时间单位）为单位的文档集合。

2. 文本的特征表示。

实现一个模块，通过一个或多个函数，将每个文档转变为词频特征表示，
以形成文档-词语的词频矩阵，可以选择使用sklearn中的CountVectorizer和TfidfVectorizer两种方式。
也可以使用gensim中的dictionary.doc2bow等。

3. 文本的话题分析。

实现一个模块，通过一个或多个函数，
借助sklearn.decomposition中的LatentDirichletAllocation构建主题模型
（话题数目可以自主指定），并对发现的主题进行分析
（每个主题对应的词语可利用model.components_来查看，每篇文档的主题概率分布可通过model.transform来查看）。
也可以参考demo里的ldav.py，用gensim进行LDA分析，并进行可视化。

4. 序列化保存。

利用pickle或json对所得到的lda模型、对应的词频矩阵、以及特征表示等进行序列化保存。
'''


import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 第一问
class ShopDataProcessor:
    def __init__(self, file_path):
        """
        初始化 ShopDataProcessor 类，用于处理商店数据。
        
        :param file_path: CSV 文件的路径。
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """从 CSV 文件加载数据。"""
        try:
            self.data = pd.read_csv(self.file_path)
            print("数据加载成功。")
        except Exception as e:
            print(f"加载数据出错: {e}")

    def aggregate_by_year(self, shop_id):
        """
        按年聚合某店铺的评论，将某年发布的所有评论视为一个文档。
        
        :param shop_id: 店铺 ID。
        :return: 按年聚合的数据 DataFrame。
        """
        if self.data is None:
            print("数据未加载。请先调用 load_data() 方法。")
            return None
        
        # 过滤出特定店铺的数据
        shop_data = self.data[self.data['shopID'] == shop_id]
        
        # 确保年份列存在
        if 'year' not in shop_data.columns:
            print("数据缺少 'year' 列，无法按年聚合。")
            return None
        
        # 按年聚合评论
        aggregated_data = shop_data.groupby('year').agg({
            'cus_comment': lambda x: ' '.join(x.fillna('').astype(str)),  # 合并评论内容，处理缺失值并转换为字符串
        }).reset_index()
        
        print(f"shopID {shop_id} 的数据已按年成功聚合。")
        return aggregated_data



# 第二问
class TermFrequencyGenerator:
    def generate_term_frequency_matrix(self, aggregated_data, text_column='cus_comment'):
        """
        生成文档-词语的词频矩阵。
        
        :param aggregated_data: 按年聚合的数据 DataFrame。
        :param text_column: 包含文本数据的列名。
        :return: 词频矩阵 DataFrame 和 CountVectorizer 对象。
        """
        if aggregated_data is None or text_column not in aggregated_data.columns:
            print("聚合数据为空或缺少指定的文本列。")
            return None, None
        
        # 初始化 CountVectorizer
        vectorizer = CountVectorizer()
        
        # 拟合并转换文本数据为词频矩阵
        term_frequency_matrix = vectorizer.fit_transform(aggregated_data[text_column])
        
        # 将稀疏矩阵转换为 DataFrame，以便查看
        term_frequency_df = pd.DataFrame(term_frequency_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=aggregated_data['year'])
        
        print("词频矩阵生成成功。")
        return term_frequency_df, vectorizer
    


# 第三问
class GensimLDAModeler:
    def __init__(self, num_topics=5):
        """
        初始化 GensimLDAModeler 类，用于训练 LDA 模型。
        
        :param num_topics: 指定的主题数目。
        """
        self.num_topics = num_topics
        self.dictionary = None
        self.corpus = None
        self.model = None

    def prepare_data(self, texts):
        """
        准备数据，将文本列表转换为词典和语料库（词袋表示）。
        
        :param texts: 文本列表，每个元素是一个分词后的文档（列表形式）。
        """
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

    def fit_lda_model(self):
        """训练 LDA 模型。"""
        self.model = gensim.models.LdaModel(
            self.corpus, 
            num_topics=self.num_topics, 
            id2word=self.dictionary, 
            passes=15,
            random_state=42
        )

    def print_top_words(self, n_top_words=5):
        """
        打印每个主题的前 n_top_words 个词语。
        
        :param n_top_words: 每个主题显示的词语数量。
        """
        for idx, topic in self.model.print_topics(-1):
            print(f"主题 {idx}: {topic}")



# 第四问
class SerializationHandler:
    @staticmethod
    def save_objects(lda_model, term_frequency_df, vectorizer, dictionary, corpus, filename_prefix):
        """
        保存 LDA 模型、词频矩阵、特征表示和语料库到磁盘。
        
        :param lda_model: 训练好的 LDA 模型。
        :param term_frequency_df: 词频矩阵（DataFrame 格式）。
        :param vectorizer: 用于生成词频矩阵的 CountVectorizer 对象。
        :param dictionary: Gensim 词典对象。
        :param corpus: Gensim 语料库（词袋表示）。
        :param filename_prefix: 保存文件的前缀，用于生成文件名。
        """
        with open(f"{filename_prefix}_lda_model.pkl", 'wb') as f:
            pickle.dump(lda_model, f)
        with open(f"{filename_prefix}_term_frequency_df.pkl", 'wb') as f:
            pickle.dump(term_frequency_df, f)
        with open(f"{filename_prefix}_vectorizer.pkl", 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(f"{filename_prefix}_dictionary.pkl", 'wb') as f:
            pickle.dump(dictionary, f)
        with open(f"{filename_prefix}_corpus.pkl", 'wb') as f:
            pickle.dump(corpus, f)

    @staticmethod
    def load_objects(filename_prefix):
        """
        从磁盘加载之前保存的 LDA 模型、词频矩阵、特征表示和语料库。
        
        :param filename_prefix: 保存文件的前缀，用于生成文件名。
        :return: 加载的 LDA 模型、词频矩阵、特征表示、词典和语料库。
        """
        with open(f"{filename_prefix}_lda_model.pkl", 'rb') as f:
            lda_model = pickle.load(f)
        with open(f"{filename_prefix}_term_frequency_df.pkl", 'rb') as f:
            term_frequency_df = pickle.load(f)
        with open(f"{filename_prefix}_vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        with open(f"{filename_prefix}_dictionary.pkl", 'rb') as f:
            dictionary = pickle.load(f)
        with open(f"{filename_prefix}_corpus.pkl", 'rb') as f:
            corpus = pickle.load(f)
        return lda_model, term_frequency_df, vectorizer, dictionary, corpus


# 第五问
class GensimLDAModeler_5:
    def __init__(self):
        self.dictionary = None
        self.corpus = None
        self.model = None
 
    def prepare_data(self, texts):
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
 
    def fit_lda_model(self, num_topics, corpus, dictionary):
        model = gensim.models.LdaModel(
            corpus, 
            num_topics=num_topics, 
            id2word=dictionary, 
            passes=15,
            random_state=42
        )
        return model
 
def compute_perplexity(model, corpus):
    """计算模型的困惑度。"""
    return model.log_perplexity(corpus)
 
def plot_perplexity(perplexities, topic_range):
    """绘制困惑度随话题数变化的曲线。"""
    plt.plot(topic_range, perplexities, marker='o')
    plt.xlabel('话题数')
    plt.ylabel('困惑度')
    plt.title('困惑度随话题数变化的曲线')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = Path("第3周\\week3.csv")
    processor = ShopDataProcessor(file_path)
    processor.load_data()
    
    shop_id = 518986
    aggregated_data_for_shop = processor.aggregate_by_year(shop_id)
    
    if aggregated_data_for_shop is not None:
        tf_generator = TermFrequencyGenerator()
        term_frequency_matrix, vectorizer = tf_generator.generate_term_frequency_matrix(aggregated_data_for_shop)
        
        texts = aggregated_data_for_shop['cus_comment'].apply(lambda x: x.split()).tolist()
        
        gensim_modeler = GensimLDAModeler(num_topics=5)
        gensim_modeler.prepare_data(texts)
        gensim_modeler.fit_lda_model()
        
        gensim_modeler.print_top_words()
        
        handler = SerializationHandler()
        handler.save_objects(
            gensim_modeler.model, 
            term_frequency_matrix, 
            vectorizer, 
            gensim_modeler.dictionary, 
            gensim_modeler.corpus, 
            'lda_model_output'
        )
        
        loaded_lda_model, loaded_term_frequency_df, loaded_vectorizer, loaded_dictionary, loaded_corpus = handler.load_objects('lda_model_output')
        
        for idx, topic in loaded_lda_model.print_topics(-1):
            print(f"加载的主题 {idx}: {topic}")

        modeler = GensimLDAModeler_5()
        modeler.prepare_data(texts)
        
        # 定义要测试的话题数范围
        topic_range = range(15, 30)  # 例如，从2到15个话题
        perplexities = []
        
        for k in topic_range:
            lda_model = modeler.fit_lda_model(k, modeler.corpus, modeler.dictionary)
            perplexity = compute_perplexity(lda_model, modeler.corpus)
            perplexities.append(perplexity)
            print(f"话题数 {k} 的困惑度: {perplexity}")
        
        rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
        rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
        
        # 绘制困惑度曲线
        plot_perplexity(perplexities, topic_range)