import datetime
import matplotlib.pyplot as plt

#注意需要安装这个包
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pickle

def sort_doc(file_path):
    '''
    文档分类
    :param file_path: 分档路径
    :return:分类结果
    '''
    # 定义时间段
    morning_start = datetime.time(4, 0, 0)
    noon_start = datetime.time(12, 0, 0)
    evening_start = datetime.time(20, 0, 0)

    # 定义数据分类字典
    data_by_time = {
        "morning": [],
        "noon": [],
        "evening": []
    }

    # 读取数据文件
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 解析每一行数据
            if line.find('cus_id')>=0:
                continue
            fields = line.strip().split(",")
            text = fields[3]
            time_str = fields[1]
            # 将时间字符串转换为datetime对象
            # 2018/9/26 9:13
            time = datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M").time()

            # 根据时间段将数据添加到相应的分类列表中
            if time >= morning_start and time < noon_start:
                data_by_time["morning"].append(text)
            elif time >= noon_start and time < evening_start:
                data_by_time["noon"].append(text)
            else:
                data_by_time["evening"].append(text)
    return data_by_time

if __name__ == '__main__':
    data_by_time = sort_doc(file_path = './data_demo.csv')
    docs = [doc for t in data_by_time for doc in data_by_time[t]]
    print(len(docs))
    #for doc in docs:
    #    print(doc)
    #exit()

    # 将文本转换成词频矩阵
    vectorizer = CountVectorizer()
    # 将文本转换成tf-idf矩阵
    #vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(docs)

    # 计算困惑度绘制elbow图确定主题数量
    perplexity_scores = []
    k_range = range(1, 6) # 假设k的范围是1到5
    for k in k_range:
        lda = LatentDirichletAllocation(n_components=k)
        lda.fit(X)
        perplexity_scores.append(lda.perplexity(X))
    plt.plot(k_range, perplexity_scores, '-o')
    plt.xlabel('Number of topics')
    plt.ylabel('Perplexity')
    plt.show()

    
    k = 5

    # 使用LatentDirichletAllocation构建主题模型
    lda = LatentDirichletAllocation(n_components=k)
    lda.fit(X)

    # 输出每个主题对应的词语
    feature_names = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        print(f"Topic {i}:")
        top_words = [feature_names[j] for j in topic.argsort()[:-6:-1]]
        print(top_words)

    # 输出每篇文档的主题概率分布
    for i in range(len(docs)):
        print(f"Document {i}:")
        print(lda.transform(X[i]))
    # 输出结果

    pickle.dump((lda, X, vectorizer), open('./lda_model.pkl','wb'))
