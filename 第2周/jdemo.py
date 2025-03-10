import jieba
import jieba.posseg as pseg
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

doc = '今天这一件是我们家的爆款爆款爆款爆款爆款'
#C:\Windows\Fonts\simfang.ttf # for windows
fpath='C:\\Windows\\Fonts\\SimHei.ttf'


terms = jieba.cut(doc)
terms = list(terms)
tfreq = Counter(terms)
for term in tfreq:
	print(f'{term}\t{tfreq[term]}')

wd = WordCloud(font_path = fpath, width = 800, height = 600)
wd.fit_words(tfreq)
#wd.to_file('./wd.png')
plt.imshow(wd)
plt.axis('off')
plt.show()