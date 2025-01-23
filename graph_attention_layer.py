"""
The implementation follows https://github.com/danielegrattarola/keras-gat/tree/master/keras_gat,
which is released under MIT License.
"""
# 这是模块导入部分，确保了一些特定的特性将从今后的 Python 版本导入，以及导入了 Keras 框架的一些组件，这主要用于定义自定义层和使用 Keras 提供的一些函数
from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU

# GraphAttention 类继承自 Keras 的 Layer 类，是定义自定义层的基础：
# 构造函数 `__init__` 定义了图注意力层的基本参数和权重初始化方法。
#
# - `F_` 是输出尺寸的大小。
# - `attn_heads` 是并行的注意力头数量。
# - `attn_heads_reduction` 决定如何整合不同注意力头的输出。有效选项："concat"（串联各个头的输出）和 "average"（取平均）。
# - 其他参数如 `dropout_rate` 等用于定义如何在层中应用正则化和权重初始化。
class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction 
        self.dropout_rate = dropout_rate  
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads
        else:
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

# `build`方法是用于创建层的权重的：
# 在 `build` 中，初始化了与图注意力机制相关的各种权重矩阵。因为这个层有多个注意力头，所以每个头都需要它自己的权重集。
    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
            
        self.built = True
# `call` 方法定义了在数据通过这层时如何计算：
#     在`call`中，进行了以下操作：
#     1.用先前创建的权重矩阵去转换节点特征。
#     2.应用注意力机制去计算每个节点对其邻居的关注度。
#     3.将经过注意力加权的邻居特征聚合起来生成新的节点表示。
#     4.如果定义了多个注意力头，还会将各个头的输出结合起来。
#     5.应用激活函数激活最终的节点表示。
    def call(self, inputs):

# 1.首先，输入被分离为特征矩阵`X`（节点的特征）和邻接矩阵`A`，然后计算它们的维度`N`（节点数量）和`F`（每个节点特征的数量）：

        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)
        N = K.shape(X)[0]
        F = K.shape(X)[1]

# 初始化一个空的列表 `outputs` 来保存每个注意力头的输出。
        outputs = []

# 对于每个注意力头：
# - 通过特征矩阵 `X` 和权重矩阵 `kernel` 计算注意力得分，注意力核 `attention_kernel` 被分为两部分：一部分计算节点自注意力得分 `attn_for_self`，另一部分计算节点对邻居注意力得分 `attn_for_neighs`。
# - 使用 Gaussian 注意力分数计算每个节点对每个邻居的注意力系数，然后应用一个掩码来保证只有真正相邻的节点之间有非零的注意力系数。
# - 在计算了被正则化（通过softmax）的注意力系数后，应用 dropout 层。这有利于防止模型在训练过程中过度依赖某些特定的节点连接。

        for head in range(self.attn_heads):
            kernel = self.kernels[head]
            F_ = K.shape(kernel)[-1]
            attention_kernel = self.attn_kernels[head]
            
            
            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            attn_for_self = K.dot(features, attention_kernel[0])    
            attn_for_neighs = K.dot(features, attention_kernel[1])  

            dense = K.exp(-K.square(attn_for_self - K.transpose(attn_for_neighs))/1e-0)
            
            mask_0 = K.abs(A)
            mask_pos = 0.5 * (A + 1.) * mask_0
            
            dense_pos = K.exp(dense) * mask_pos
            dense_pos = K.transpose(K.transpose(dense_pos) / K.sum(dense_pos, axis=1))
            dense = dense_pos

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # 计算节点特征的加权和，使用注意力分数和邻居节点特征，如果使用偏置，则将其添加到这个和中。
            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # 最终添加每个头计算后的特征到`outputs`列表
            # Add output of attention head to final output
            outputs.append(node_features)


        # 根据定义的注意力头整合方法（`concat` 或 `average`），整合每个头的输出。`concat` 方法会增加特征维度，而 `average` 则保持不变
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

# 激活函数被应用到输出上以引入非线性
        output = self.activation(output)
        return output
# `compute_output_shape` 方法计算这个层的输出形状：
    # 这个函数基于输入形状和定义的参数（如注意力头的数量和减少方法）计算输出的形状。
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
# `compute_output_shape` 方法根据输入形状和层的参数输出最终结果的形状。
# 当 `attn_heads_reduction` 为 `concat` 时，输出维数是注意力头数乘以每个头的输出维数 `F_`。若为 `average`，输出维度不变。