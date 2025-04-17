from sklearn.metrics.pairwise import cosine_similarity
import numpy as np #继续接触并使用

from ali_embeddings import v1,v2,v3

# 假设有两个向量
# cosine_similarity 要求输入为 二维数组，即 (n_samples, n_features) 的形状，因此要 .reshape(1, -1)。
vec1 = np.array(v1).reshape(1, -1)
vec2 = np.array(v2).reshape(1, -1)
vec3 = np.array(v3).reshape(1, -1)

# 计算余弦相似度
similarity = cosine_similarity(vec1, vec2)

#输出是一个矩阵,如果只比较两个向量，结果是 1x1 矩阵，所以取 [0][0]。
print("余弦相似度(test.jpg, test2.jpg):", similarity[0][0])

similarity_2 = cosine_similarity(vec1, vec3)
print("余弦相似度(test.jpg, test3.jpeg):", similarity_2[0][0])