import jieba
import jieba.posseg as pseg
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stopwords_path = 'C:\\Users\\13261\\Desktop\\大二\\大二下课程内容\\Python\\pythonDataAnalysis\\第2周\\cn_stopwords.txt'
path = 'C:\\Users\\13261\\Desktop\\大二\\大二下课程内容\\Python\\pythonDataAnalysis\\第2周\\week2.txt'
fpath='C:\\Windows\\Fonts\\SimHei.ttf'

# 第一问使用的函数
def Q1(file_name):
    with open(file_name, 'r', encoding='UTF-8') as file:
        for i in range(10):
            line = file.readline()
            if not line:
                break  # 文件行数不足10行，提前结束
            print(line, end='')

# 调用第一问的函数
Q1(path)

print("\n\n")

# 第二问开始
with open(path, 'r', encoding='UTF-8') as file:
    # 读取整个文件内容
    content = file.read()
 
    # 分词
    terms = jieba.cut(content)
    terms = list(terms)
 
    # 统计词频
    tfreq = Counter(terms)
 
    # 打印词频
    # for term in tfreq:
        # print(f'{term}\t{tfreq[term]}')

print("\n\n")

# 第三问开始
sorted_tfreq = sorted(tfreq.items(), key = lambda item :item[1], reverse = True)
dic_sortde_tfreq = dict(sorted_tfreq)

i = 0

for term in dic_sortde_tfreq:
    print(f'{term}\t{dic_sortde_tfreq[term]}')
    i += 1
    if i >= 10:
            break


print("\n\n")

# 第四问开始
stop_words = set()
with open(stopwords_path, 'r', encoding='UTF-8') as f:
    for line in f:
        stop_words.add(line.strip())

filtered_words = {word: freq for word, freq in dic_sortde_tfreq.items() if word not in stop_words and word not in {'\n', ' '}}

i = 0

for term in filtered_words:
    print(f'{term}\t{filtered_words[term]}')
    i += 1
    if i >= 10:
            break

print("\n\n")

# 第五问开始

# 创建 WordCloud 对象，指定字体路径
wd = WordCloud(font_path=fpath)

# 使用过滤后的词频字典生成词云
wd.fit_words(filtered_words)

# 将生成的词云保存为图像文件 'wd.png'
# wd.to_file('./wd.png')

# 使用 matplotlib 显示词云图像
plt.imshow(wd)

# 隐藏坐标轴，使图像更清晰
plt.axis('off')

# 显示图像
plt.show()

print("\n\n")
#附加部分

#第六问开始

# 用于存储不同词性的频率，键为词性，值为频数
pos_freq = {}
 
# 对每个词进行词性标注
for word, freq in filtered_words.items():
    words = pseg.cut(word)
    for w, p in words:
        if w == word:  # 确保分词结果与原词匹配
            if p in pos_freq:
                pos_freq[p] += freq
            else:
                pos_freq[p] = freq
 
# 输出不同词性的频率
print("不同词性的频率：")
for pos, freq in pos_freq.items():
    print(f"{pos}: {freq}")
 
# 选择一个特定词性进行词云展示，比如名词 'n'
selected_pos = 'n'
selected_words = {word: freq for word, freq in filtered_words.items() 
                  if any(p == selected_pos for w, p in pseg.cut(word))}
 
# 生成词云
wordcloud = WordCloud(font_path=fpath,
                      width=800, 
                      height=400, 
                      background_color='white').generate_from_frequencies(selected_words)
 
# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

print("\n\n")

# 第七问开始

# 按行读取文本文件
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# 生成 bigram，bigram指的是两个连续在一起的词
def generate_bigrams(sentences):
    bigrams = []
    for sentence in sentences:
        words = list(jieba.cut(sentence.strip()))
        # 创建 bigram
        bigrams.extend([(words[i], words[i+1]) for i in range(len(words)-1)])
    return bigrams

# 统计 bigram 频率
def count_bigrams(bigrams):
    return Counter(bigrams)

# 可视化高频 bigram
def visualize_bigrams(bigram_freq, top_n=20):
    # 将 bigram 转换为字符串格式，以便在词云中使用
    bigram_str_freq = {f'{w1} {w2}': freq for (w1, w2), freq in bigram_freq.items()}
    
    # 生成词云
    wordcloud = WordCloud(
        font_path=fpath,
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(bigram_str_freq)
    
    # 可视化显示所有bigram的词云
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 主流程
file_path = path
sentences = read_text_file(file_path)
bigrams = generate_bigrams(sentences)
bigram_freq = count_bigrams(bigrams)

# 输出前 10 个高频 bigram
print("前 10 个高频 bigram：")
for bigram, freq in bigram_freq.most_common(10):
    print(f"{bigram}: {freq}")

# 可视化高频 bigram
visualize_bigrams(bigram_freq, top_n=20)