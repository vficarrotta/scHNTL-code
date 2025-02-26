"""
The implementation of Network Enhancement (NE) is modified from
https://github.com/wangboyunze/Network_Enhancement on 2021.11.11, 
which is released under GNU General Public License v3.0. 
"""
# 这一行较少在现代 Python 代码中出现，因为它主要用于 Python 2.x 版本来实现 Python 3.x 版式的打印函数。
from __future__ import print_function

# 这些行导入了几个标准库：`os` 用于操作与操作系统交互的功能，如文件路径操作；
# `pickle` 用于序列化与反序列化对象；`sys` 提供对与 Python 解释器紧密相关的变量和函数的访问；`time` 用于获取当前时间和日期信息。
import os
import pickle as pkl
import sys
import time

# 这些行导入了科学计算库 numpy，数据结构与数据分析工具包 pandas，以及 scipy 中的稀疏矩阵模块。
import numpy as np
import pandas as pd
import scipy.sparse as sp


# 这些行导入了 scikit-learn 机器学习库中的一些模块，
# 用于进行 t-分布随机邻域嵌入（t-SNE），主成分分析（PCA），K-均值聚类算法，度量和评价方法，以及 SciPy 统计模块中的斯皮尔曼等级相关计算。
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA # PCA IS CALLED IN THIS FILE; ADD GLMPCA HERE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats import spearmanr

# GK = pd.read_csv('D:\\学习\\研究生学习\\论文\\bioinfomatics\\scGAC a graph attentional architecture for clustering\\github\\scGAC改进本地\\data\\Yan\\Kernel_matrix.tsv', sep='\t', header=None).values
# 这是一个归一化函数，将输入的特征值转化为每行和为100000的比例数据，并应用对数变换，以让特征值更适合后续分析。
def normalization(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features

# 这个函数与 `normalization` 类似，但是乘的常数为1000000，针对 NE 的特征归一化。
def normalization_for_NE(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 1000000
    features = np.log2(features + 1)
    return features

# 这个函数实现了 NE 算法的一部分，其中 w 是权重矩阵，N 是网络的大小，eps 是为了防止除以0添加的小量。
def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn

# 这个函数用于确定一个亲和力矩阵的支配集，即筛选与每个点最关联的 NR_OF_KNN 个点。
def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix

# 这个函数计算转换场矩阵，也是 NE 算法的一部分。
def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W

# 这个函数根据输入（预处理后的亲和力矩阵）计算经过网络增强后的矩阵。
def getNeMatrix(W_in):
    N = len(W_in)

    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W


"""
Construct a graph based on the cell features
"""
# 这个函数基于输入数据集的特征构造一个图，可以使用不同的方法来计算特征间的相关性，比如皮尔森、斯皮尔曼或 NE。


def getGraph(dataset_str, features, L, K, method):# 定义函数getGraph，它接受五个参数：dataset_str（数据集名称），features（特征矩阵），L和K（用于确定邻接矩阵的阈值），以及method（计算相关性的方法
    print(method)

    Graph_Matrix = np.corrcoef(features)
    if method == 'pearson':# 如果方法是皮尔逊相关性，那么就计算特征的皮尔逊相关性矩阵。
        co_matrix = np.corrcoef(features)
    elif method == 'spearman':# 如果方法是斯皮尔曼相关性，那么就计算特征的斯皮尔曼相关性矩阵。
        co_matrix, _ = spearmanr(features.T)
    elif method == 'NE':# 如果方法是’NE’，那么首先计算特征的皮尔逊相关性矩阵
        co_matrix = np.corrcoef(features)
        # print(np.corrcoef(features))
        #co_matrix = co_matrix/2

        # 然后，检查是否已经计算过NE矩阵并保存在文件中。如果是，那么就直接读取。否则，对特征进行归一化，计算归一化特征的皮尔逊相关性矩阵，然后使用getNeMatrix函数计算NE矩阵，并将其保存到文件中。
        NE_path = 'result/NE_' + dataset_str + '.csv'
        if os.path.exists(NE_path):
            NE_matrix = pd.read_csv(NE_path).values
        else:
            features = normalization_for_NE(features)
            in_matrix = np.corrcoef(features)
            NE_matrix = getNeMatrix(in_matrix)
            pd.DataFrame(NE_matrix).to_csv(NE_path, index=False)

# 设置NE矩阵对角线上的元素为该行最大值的sim_sh倍。
        N = len(co_matrix)
        sim_sh = 1.
        for i in range(len(NE_matrix)):
            NE_matrix[i][i] = sim_sh * max(NE_matrix[i])
        
        data = NE_matrix.reshape(-1)
        data = np.sort(data)
        data = data[:-int(len(data)*0.02)]
        
        min_sh = data[0]
        max_sh = data[-1]
        
        delta = (max_sh - min_sh) / 100

    # 将数据分成20个区间，并计算每个区间的中值和元素数量。
        temp_cnt = []
        for i in range(20):
            s_sh = min_sh + delta * i
            e_sh = s_sh + delta
            temp_data = data[data > s_sh]
            temp_data = temp_data[temp_data < e_sh]
            temp_cnt.append([(s_sh + e_sh)/2, len(temp_data)])

        # 寻找第一个局部最小值，如果找到，就将其设置为候选阈值。
        candi_sh = -1
        for i in range(len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if 0 < i < len(temp_cnt) - 1:
                if pear_cnt < temp_cnt[i+1][1] and pear_cnt < temp_cnt[i-1][1]:
                    candi_sh = pear_sh
                    break

        # 如果没有找到局部最小值，那么就寻找第一个元素数量小于前一个区间元素数量一半的区间，如果找到，就将其设置为候选阈值。
        if candi_sh < 0:
            for i in range(1, len(temp_cnt)):
                pear_sh, pear_cnt = temp_cnt[i]
                if pear_cnt * 2 < temp_cnt[i-1][1]:
                    candi_sh = pear_sh
        # 如果还是没有找到候选阈值，那么就将其设置为0.3。
        if candi_sh == -1:
            candi_sh = 0.3

        # 计算NE矩阵中小于等于候选阈值的元素的比例，然后根据这个比例计算真正的阈值。然后，将相关性矩阵中对应的元素设置为0。
        propor = len(NE_matrix[NE_matrix <= candi_sh])/(len(NE_matrix)**2)
        propor = 1 - propor
        thres = np.sort(NE_matrix)[:, -int(len(NE_matrix)*propor)]
        co_matrix.T[NE_matrix.T <= thres] = 0

    # 如果方法既不是’pearson’，也不是’spearman’，也不是’NE’，那么就直接返回。
    else:
        return

    # 计算相关性矩阵的大小。
    N = len(co_matrix)

    # 计算每行最大的K个元素。
    up_K = np.sort(co_matrix)[:,-K]

    # 生成一个新的矩阵，其中大于等于每行最大的K个元素的元素设置为1，其他元素设置为0。
    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1

    # 计算整个相关性矩阵中最大的L个元素，然后将新矩阵中小于这个阈值的元素设置为0。
    thres_L = np.sort(co_matrix.flatten())[-int(((N*N)//(1//(L+1e-8))))]
    mat_K.T[co_matrix.T < thres_L] = 0

    # Graph_Matrix *= mat_K
    return mat_K, Graph_Matrix


"""
Load scRNA-seq data set and perfrom preprocessing
"""
# 这个函数用于加载和预处理 scRNA-seq 数据集，其中也包括通过 PCA 进行维度减少。
def load_data(data_path, dataset_str, PCA_dim, is_NE=True, n_clusters=20, K=None):
    # Get data
    DATA_PATH = data_path

    data = pd.read_csv(DATA_PATH, index_col=0, sep='\t')
    cells = data.columns.values
    genes = data.index.values
    features = data.values.T

    # Preprocess features
    features = normalization(features)

    # Construct graph
    N = len(cells)
    avg_N = N // n_clusters
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 6)

    L = 0
    if is_NE:
        method = 'NE'
    else:
        method = 'pearson'
    adj, result= getGraph(dataset_str, features, L, K, method)

    # feature tranformation
    if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
        pca = PCA(n_components = PCA_dim)                                # PCA call change to GLMPCA, adjust downstream auxiliary functions and operations as necessary
        features = pca.fit_transform(features)
    else:
        var = np.var(features, axis=0)
        min_var = np.sort(var)[-1 * PCA_dim]
        features = features.T[var >= min_var].T
        features = features[:, :PCA_dim]
    print('Shape after transformation:', features.shape)
    
    features = (features - np.mean(features)) / (np.std(features))
  
    return adj, features, cells, genes, result, K

# 这个函数将聚类结果保存到文件中。
def saveClusterResult(y_pred, cells, dataset_str):
    pred_path = 'result/pred_'+dataset_str+'.txt'

    result = []
    for i in range(len(y_pred)):
        result.append([cells[i], y_pred[i]])
    result = pd.DataFrame(np.array(result), columns=['cell', 'label'])
    result.to_csv(pred_path, index=False, sep='\t')

# 最后，这个函数应用 K-均值聚类算法到数据上，并返回聚类标签和中心。
def my_kmeans(K, hidden, dataset_str):
    print('--------------------------------')
    print('Original data shape:', hidden.shape)

    pred_path = 'pred/pretrain_'+dataset_str+'.txt'

    kmeans = KMeans(n_clusters=K, random_state=0).fit(hidden)
    labels = kmeans.labels_

    print('Kmeans end')
    print('--------------------------------')
    return kmeans.labels_, kmeans.cluster_centers_

from sklearn.cluster import KMeans

def my_kmeans2(K, hidden, dataset_str):
    # 第一阶段: 将数据聚成20K个簇
    kmeans_20K = KMeans(n_clusters=20*K, random_state=0).fit(hidden)
    centers_20K = kmeans_20K.cluster_centers_  # 获取20K个簇的中心

    # 第二阶段: 对这20K个簇的中心进行K-means聚类，聚成K个簇
    kmeans_K = KMeans(n_clusters=K, random_state=0).fit(centers_20K)
    final_centers = kmeans_K.cluster_centers_  # 获取最终K个簇的中心

    # 为原始数据点分配到这K个簇中
    labels_final = kmeans_K.predict(hidden)

    return labels_final, final_centers
