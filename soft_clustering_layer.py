from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
from keras.layers import Layer, InputSpec

import tensorflow as tf

# 定义 `ClusteringLayer` 类，继承自 `Layer` 类，这是构建自定义 Keras 层的基础。
class ClusteringLayer(Layer):
    # 构造函数`__init__`
    # 初始化聚类层的一些基本参数。
    # - `n_clusters`: 聚类中心的数量。
    # - `N`: 输入数据的样本数量。
    # - `sh`: 平滑参数，帮助适度抑制聚类的概率以避免过度置信。
    # - `mode`: 模式选择，通常用于决定如何重新计算聚类的概率。
    # - `weights`: 可以提供聚类中心的初始权重。
    # - `alpha`: 概率赋值的分母平滑系数。

    def __init__(self,
                 n_clusters,
                 N,
                 mode='q2',
                 sh = .1,
                 weights=None,
                 alpha=1.0,
                 **kwargs):
        self.n_clusters = n_clusters
        self.N = N
        self.sh = sh
        self.mode = mode
        self.alpha = alpha
        self.clusters = weights

        super(ClusteringLayer, self).__init__(**kwargs)

    # `build`方法确保了输入的维度是两维的，会在这个方法里初始化一些与输入形状相关的权重，
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.built = True

    # `call`方法是层的核心，定义了当数据通过该层时如何进行计算。
    # 1.首先计算输入和聚类中心间的距离。
    # 2.根据这些距离，计算每个点属于每个聚类中心的软分配概率`q`。
    # 3.`q_ik`是每个点分配给聚类中心的概率，通过归一化所有点到每个中心的距离的倒数来计算。
    # 4.`q_`再次对这些概率进行处理，可以选择性地进行平方处理并进行另一次归一化。
    # 5.接着确定并移除每个点的最高概率以外的值。
    # 6.应用`ReLU`函数，加上一个平滑的阈值`sh`来调整概率。
    # 7.最后，通过输入和概率来更新聚类中心。

    def call(self, inputs, **kwargs):
        # q_ik = (1 + ||z_i - miu_k||^2)^-1
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        # q_ik = q_ik / sigma_k' q_ik'
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
            
        # q'_ik = q_ik ^ 2 / sigma_i q_ik
        if self.mode == 'q2':
            q_ = q ** 2 / K.sum(q, axis=0)
            q_ = K.transpose(K.transpose(q_) / K.sum(q_, axis=1))
        else:
            q_ = q + 1e-20
        
        q_idx = K.argmax(q_, axis=1)
        q_mask = K.one_hot(q_idx, self.n_clusters)
        # q'_ik = 0 if q'_ik < max(q'_i)
        q_ = q_mask * q_
         
        q_ = K.relu(q_ - self.sh)
        q_ = q_ + K.sign(q_) * self.sh
        # miu_k = sigma_i q'_ik * z_i
        self.clusters = K.dot(K.transpose(q_ / K.sum(q_, axis=0)), inputs)
       
        return q

    # `compute_output_shape`方法返回输出的形状。此方法告诉Keras该层的输出是如何形成的，以保证模型的其他部分正确地处理这个形状。
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
# 总的来说，`ClusteringLayer` 层是用来根据特征数据进行聚类的一个算法的 Keras 表示。该层的输出是每个数据点属于每个聚类中心的概率，
# 该层在更新自己的状态时也会更新聚类中心的位置。这使得它个特别适合于深度聚类任务，其中我们希望网络自己学习如何根据数据本身的结构将点归纳到聚类中。