import csv
from collections import defaultdict, Counter
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Q1
'''
EMOTION_LEXICON_PATH = '第3周\emotion_lexicon'

def load_emotion_lexicon():
    """
    加载情绪字典，返回一个字典，其中键是情绪类型，值是对应情绪词的集合。
    """
    emotion_dict = {
        'anger': set(),
        'disgust': set(),
        'fear': set(),
        'sadness': set(),
        'joy': set()
    }
    
    for emotion in emotion_dict:
        file_path = os.path.join(EMOTION_LEXICON_PATH, f'{emotion}.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            emotion_dict[emotion] = set(word.strip() for word in f)
    
    return emotion_dict

# 使用闭包来确保情绪字典只加载一次
def create_sentiment_analyzer():
    """
    创建情绪分析器，返回两个函数：混合情绪分析和唯一情绪分析。
    """
    emotion_dict = load_emotion_lexicon()  # 加载情绪字典，只加载一次
    
    def mixed_sentiment_analysis(comment):
        """
        对单条评论进行混合情绪分析，返回每种情绪的比例。
        
        参数:
            comment (str): 评论内容。
        
        返回:
            dict: 每种情绪的比例。
        """
        emotion_count = defaultdict(int)  # 用于统计每种情绪词的出现次数
        words = comment.split()  # 将评论分词
        total_emotion_words = 0  # 总情绪词数
        
        for word in words:
            for emotion, lexicon in emotion_dict.items():
                if word in lexicon:
                    emotion_count[emotion] += 1
                    total_emotion_words += 1
        
        if total_emotion_words == 0:
            # 如果评论中没有情绪词，返回所有情绪的比例为0
            return {emotion: 0 for emotion in emotion_dict}
        
        # 计算每种情绪的比例
        emotion_ratio = {emotion: count / total_emotion_words for emotion, count in emotion_count.items()}
        # 确保所有情绪都有值，即使某些情绪没有出现
        for emotion in emotion_dict:
            emotion_ratio.setdefault(emotion, 0)
        
        return emotion_ratio
    
    def unique_sentiment_analysis(comment):
        """
        对单条评论进行唯一情绪分析，返回出现次数最多的情绪。
        
        参数:
            comment (str): 评论内容。
        
        返回:
            str: 主要情绪，如果有多个情绪出现次数相同，返回'ambiguous'。
        """
        emotion_count = Counter()  # 用于统计每种情绪词的出现次数
        words = comment.split()  # 将评论分词
        
        for word in words:
            for emotion, lexicon in emotion_dict.items():
                if word in lexicon:
                    emotion_count[emotion] += 1
        
        if not emotion_count:
            # 如果评论中没有情绪词，返回'neutral'
            return 'neutral'
        
        # 找到出现次数最多的情绪
        max_emotion = max(emotion_count, key=emotion_count.get)
        
        # 处理不同情绪情绪词出现次数相同的情况
        max_count = emotion_count[max_emotion]
        candidates = [emotion for emotion, count in emotion_count.items() if count == max_count]
        
        # 如果有多个候选，返回'ambiguous'，否则返回主要情绪
        return max_emotion if len(candidates) == 1 else 'ambiguous'
    
    return mixed_sentiment_analysis, unique_sentiment_analysis


mixed_analysis, unique_analysis = create_sentiment_analyzer()


'''
# 读取week3.csv文件，进行处理，并输出新的CSV文件
input_file = '第3周\week3.csv'
output_file = 'sentiment_analysis.csv'
 
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['mix', 'unique']  # 新增两列
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()  # 写入表头
    
    for row in reader:
        comment = row['cus_comment']  # 假设评论内容列名为'评论内容'
        
        # 进行情绪分析
        mix_result = mixed_analysis(comment)
        unique_result = unique_analysis(comment)
        
        # 将情绪分析结果转换为字符串形式，便于写入CSV
        # 对于mix_result，我们可以将其转换为"anger:0.2,disgust:0.1,..."的形式
        mix_str = ','.join(f"{emotion}:{ratio:.2f}" for emotion, ratio in mix_result.items())
        
        # 写入新的行，包含原始数据和情绪分析结果
        row['mix'] = mix_str
        row['unique'] = unique_result
        writer.writerow(row)
 
print(f"情绪分析结果已保存到 {output_file}")
'''

'''
Q2
'''

def time_sentiment_proportion(shopID, time, sentiment, data):
    filtered_data = data[data['shopID'] == shopID]
    proportion_data = filtered_data.groupby(time)['unique'].apply(
        lambda x: (x == sentiment).sum() / len(x) if len(x) > 0 else 0
    ).reset_index(name='proportion')
    return proportion_data

def plot_sentiment_proportion(proportion_data, time, sentiment):
    plt.figure(figsize=(10, 6))
    plt.plot(proportion_data[time], proportion_data['proportion'], marker='o')
    plt.title(f'{sentiment} Sentiment Proportion Over {time.capitalize()} for Shop {proportion_data["shopID"].iloc[0] if "shopID" in proportion_data.columns else "Unknown"}')
    plt.xlabel(time.capitalize())
    plt.ylabel(f'{sentiment} Sentiment Proportion')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 调用函数
df = pd.read_csv('sentiment_analysis.csv')
shopID = 518986
time = 'month'
sentiment = 'joy'
proportion_data = time_sentiment_proportion(shopID, time, sentiment, df)
print(proportion_data)

# 绘制图表
plot_sentiment_proportion(proportion_data, time, sentiment)


'''
Q3
'''

# 按 'unique' 列分组，并计算每组 'star' 的平均值和方差
grouped_data = df.groupby('unique')['stars'].agg(['mean', 'var']).reset_index()
 
# 打印结果以进行验证
print(grouped_data)
 
# 绘制统计图
fig, ax1 = plt.subplots(figsize=(10, 6))
 
# 绘制平均值条形图
ax1.bar(grouped_data['unique'], grouped_data['mean'], color='b', alpha=0.6, label='Mean Star Rating')
ax1.set_xlabel('Unique Sentiment')
ax1.set_ylabel('Mean Star Rating', color='b')
ax1.tick_params(axis='y', labelcolor='b')
 
# 创建一个第二个y轴，共享x轴
ax2 = ax1.twinx()
 
# 绘制方差误差条（这里用方差的值作为误差条的大小，实际中可能需要根据需要调整表示方式）
# 为了更直观地展示，我们可以将方差转换为标准差（方差的平方根），但这里仍用方差展示
ax2.errorbar(grouped_data['unique'], grouped_data['mean'], 
             yerr=np.sqrt(grouped_data['var']),  # 使用标准差作为误差条大小
             fmt='o', color='r', label='Standard Deviation (from Variance)')
ax2.set_ylabel('Standard Deviation (from Variance)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
 
# 添加图例
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
 
# 设置标题和网格
plt.title('Mean and Variance of Star Ratings by Sentiment')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
 
# 显示图形
plt.show()