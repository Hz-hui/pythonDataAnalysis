#LDA可视化

import gensim
import collections
import sys

from gensim.corpora.dictionary import Dictionary

import pyLDAvis
import pyLDAvis.gensim 

#在模块里的可执行语句 
pyLDAvis.enable_notebook()


def load_id2word_corpus(path):
    texts = []
    with open(path,'r') as f:
        for line in f:
            if line.find('cus_id')>=0:
                continue
            fs = line.strip().split(',')
            text = []
            for t in fs[3].split(' '):
                if len(t)>1:#只选长度大于1的词，类似除去停用词
                    text.append(t)
            texts.append(text)
    #print(len(texts))
    #print(texts[:10])
    #构建词典，就是word to id
    dictionary = Dictionary(texts)
    #用词典重新表达文档集
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print(common_corpus[:5])
    #构建id to word的映射，便于结果的输出
    idtow = {}
    for token in dictionary.token2id:
        idtow[dictionary.token2id[token]] = token
    return corpus, idtow, dictionary



def lda_vis(corpus, itof, tnum = 10):
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                           id2word = itof,
                                           num_topics = tnum, 
                                           alpha='auto',
                                           per_word_topics=True)
    # 输出每个话题的关键词
    print(lda_model.print_topics())
    return lda_model

if __name__ == '__main__':
    c, itw, dictionary = load_id2word_corpus(sys.argv[1])
    lda_model = lda_vis(c, itw)
    vis = pyLDAvis.gensim.prepare(lda_model, c, dictionary)