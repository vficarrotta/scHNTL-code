from __future__ import division
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

import pickle as pkl
import numpy as np
import pandas as pd
import sys
import time
import os
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy
import scipy.stats
import tensorflow as tf

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras import backend as K
import keras

from graph_attention_layer import GraphAttention
from soft_clustering_layer import ClusteringLayer
from utils import load_data, my_kmeans, saveClusterResult, my_kmeans2

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('dataset_str', type=str, help='name of dataset')
parser.add_argument('n_clusters', type=int, help='expected number of clusters')
parser.add_argument('--subtype_path', default=None, type=str, help='path of true labels for evaluation of ARI and NMI')
parser.add_argument('--k', default=None, type=int, help='number of neighbors to construct the cell graph')
parser.add_argument('--is_NE', default=True, type=bool, help='use NE denoise the cell graph or not')
parser.add_argument('--PCA_dim', default=512, type=int, help='dimensionality of input feature matrix that transformed by PCA')
parser.add_argument('--F1', default=64, type=int, help='number of neurons in the 1-st layer of encoder')
parser.add_argument('--F2', default=16, type=int, help='number of neurons in the 2-nd layer of encoder')
parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate of neurons in autoencoder')
parser.add_argument('--l2_reg', default=0, type=float, help='coefficient for L2 regularizition')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for training')
parser.add_argument('--pre_lr', default=2e-4, type=float, help='learning rate for pre-training')
parser.add_argument('--pre_epochs', default=100, type=int, help='number of epochs for pre-training')
parser.add_argument('--epochs', default=2000, type=int, help='number of epochs for pre-training')
parser.add_argument('--c1', default=1, type=float, help='weight of reconstruction loss')
parser.add_argument('--c2', default=1, type=float, help='weight of clustering loss')
parser.add_argument('--c3', default=1, type=float, help='weight of triplet loss')

args = parser.parse_args()

if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('result/'):
    os.makedirs('result/')

dataset_str = args.dataset_str
n_clusters = args.n_clusters
if args.k == 1:
    dropout_rate = 0. # To avoid absurd results
else:
    dropout_rate = args.dropout_rate

# Paths
data_path = 'data/'+dataset_str+'/data.tsv'
GAT_autoencoder_path = 'logs/GATae_'+dataset_str+'.h5'
model_path = 'logs/model_'+dataset_str+'.h5'
pred_path = 'result/pred_'+dataset_str+'.txt'
intermediate_path = 'logs/model_'+dataset_str+'_'

print(data_path, dataset_str, args.PCA_dim, args.is_NE, n_clusters, args.k)

# Read data
start_time = time.time()
A, X, cells, genes, result1, K1 = load_data(data_path, dataset_str,
                               args.PCA_dim, args.is_NE, n_clusters, args.k)
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-process: run time is %.2f '%run_time, 'minutes')


def calculate_second_order_neighbors(A):
    # Ensure A_prime is of float type
    A_prime = A.astype(float).copy()
    # Initialize identity matrix I
    I = np.eye(A.shape[0])

    # Compute A^2
    A2 = np.linalg.matrix_power(A, 2)
    # Set all elements in A^2 greater than 0 to 1
    A2[A2 > 0] = 1
    # Compute B (second-order neighbors)
    B = A2 - I - A
    # Ensure there are no negative values in B
    B[B < 0] = 0
    # Apply the bitwise AND operation with A
    B_and_A = np.multiply(B, A)

    # Calculate A'
    A_prime += 0.5 * (B - B_and_A)

    return A_prime

A15 = calculate_second_order_neighbors(A)

def calculate_negative_example_matrix(A_prime):
    # Calculate the negative example matrix
    negative_example_matrix = np.where(A_prime == 0, 1, 0)
    return negative_example_matrix

B = calculate_negative_example_matrix(A15)

# Parameters
N = X.shape[0] #Number of nodes in the graph
F = X.shape[1] #Original feature dimension

# Loss functions
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true))

def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    return loss_E

def pred_loss(y_true, y_pred):
    return y_pred

def pred_loss1(y_true, y_pred):
    # 计算 y_pred 的协方差矩阵
    y_pred_mean = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    y_pred_centered = y_pred - y_pred_mean
    covariance_matrix = tf.matmul(y_pred_centered, y_pred_centered, transpose_b=True)

    # 计算 y_pred 行的方差
    variance = tf.reduce_sum(tf.square(y_pred_centered), axis=1, keepdims=True)

    # 计算相关系数矩阵
    correlation_matrix = covariance_matrix / tf.sqrt(tf.matmul(variance, variance, transpose_b=True))
    # 根据相关系数矩阵计算正样本和负样本的距离
    positive_distances = A15 * correlation_matrix
    negative_mask = B
    negative_distances = negative_mask * correlation_matrix

    # 计算平均距离
    positive_avg_distances = tf.reduce_mean(positive_distances, axis=1)
    negative_avg_distances = tf.reduce_mean(negative_distances, axis=1)

    # 计算损失
    num_rows = tf.cast(tf.shape(y_pred)[0], tf.float32)
    triplet_loss = tf.reduce_sum(tf.maximum(0.0, -positive_avg_distances + negative_avg_distances + 0.2)) / (
                num_rows)

    return triplet_loss

# Model definition
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(args.F2,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
graph_attention_3 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout3, A_in])

dropout4 = Dropout(dropout_rate)(graph_attention_3)
graph_attention_4 = GraphAttention(F,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout4, A_in])

# Build GAT autoencoder model
GAT_autoencoder = Model(inputs=[X_in, A_in], outputs=graph_attention_4)
optimizer = Adam(lr=args.pre_lr)
GAT_autoencoder.compile(optimizer=optimizer,
              loss=maie_class_loss)
#GAT_autoencoder.summary()

# Callbacks
es_callback = EarlyStopping(monitor='loss', min_delta=0.1, patience=50)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint(GAT_autoencoder_path,
                              monitor='loss',
                              save_best_only=True,
                              save_weights_only=True)

# Train GAT_autoencoder model
start_time = time.time()
GAT_autoencoder.fit([X, A], X, epochs=args.pre_epochs, batch_size=N,
                    verbose=0, shuffle=False, callbacks=[es_callback, tb_callback, mc_callback])
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-train: run time is %.2f '%run_time, 'minutes')


# Construct a model for hidden layer
hidden_model = Model(inputs=GAT_autoencoder.input, outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)

# Get k-means clustering results of hidden representation of cells
y_pred, pre_centers = my_kmeans(n_clusters, hidden, dataset_str)
y_pred_last = np.copy(y_pred)


# Add the soft_clustering layer
soft_cluster_layer = ClusteringLayer(n_clusters,
                                     N,
                                     'q',
                                     0,
                                     pre_centers,
                                     name='clustering')(dropout3)


# Construct total model
model = Model(inputs=[X_in, A_in],
              outputs=[graph_attention_4,
                       soft_cluster_layer,
                       graph_attention_2])


optimizer = Adam(lr=args.learning_rate)
model.compile(optimizer=optimizer,
              loss=[maie_class_loss, 'kld', pred_loss1],
                    loss_weights=[args.c1, args.c2, args.c3])


# Train model
start_time = time.time()

tol = 1e-5
loss = 0

sil_logs = []
update_interval = 2
res_ite = 0
final_pred = None
max_sil = 0

latest_loss = 0
loss_logs = []
latest_loss_logs = []
min_loss = float('inf')

for ite in range(args.epochs + 1):
    if ite % update_interval == 0:
        res_ite = ite

        _, q, hid = model.predict([X, A], batch_size=N, verbose=0)

        hidden = hid.astype(float)
        hidden = pd.DataFrame(hidden)

        p = q ** 2 / q.sum(0)
        p = (p.T / p.sum(1)).T
        y_pred = q.argmax(1)

        # result_df = pd.DataFrame({'cell': cells, 'label': y_pred})
        #
        # true_path = args.subtype_path
        # true = pd.read_csv(true_path, sep='\t').values
        # true = true[:, -1].astype(int)
        #
        # ARI = adjusted_rand_score(y_pred, true)
        # NMI = metrics.normalized_mutual_info_score(true, y_pred)

        sil_hid = metrics.silhouette_score(hid, y_pred, metric='euclidean')
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        print('Iter:', ite,
              ', sil_hid:', np.round(sil_hid, 3),
              ', delta_label', np.round(delta_label, 3),
              ', loss:', np.round(loss, 2),
              )

        sil_logs.append(sil_hid)
        arr_sil = np.array(sil_logs)

        if sil_hid >= max_sil:
            final_pred = y_pred
            max_sil = sil_hid

        if len(arr_sil) >= 20 * 2:
            mean_0_n = np.mean(arr_sil[-20:])
            mean_n_2n = np.mean(arr_sil[-40: -20])
            if mean_0_n - mean_n_2n <= 0.02:
                print('Stop early at', ite, 'epoch')
                break

        if len(arr_sil) >= 3:
            if arr_sil[-2] - arr_sil[-1] >= 0.1:
                print('Stop early at', ite, 'epoch')
                break

        loss_logs.append(loss)
        arr_loss = np.array(loss_logs, dtype=object)
        latest_loss_logs.append(latest_loss)
        arr_latest_loss = np.array(latest_loss_logs, dtype=object)

        if isinstance(loss, list):
            latest_loss = loss[-1]

    loss = model.train_on_batch(x=[X, A], y=[X, p, hid])

#model.save_weights(model_path)

end_time = time.time()
run_time = (end_time - start_time) / 60
print('Train: run time is %.2f '%run_time, 'minutes')

if args.subtype_path:
    true_path = args.subtype_path
    true = pd.read_csv(true_path, sep='\t').values
    cells = true[:, 0]
    true = true[:, -1].astype(int)
    ARI = adjusted_rand_score(y_pred, true)
    NMI = metrics.normalized_mutual_info_score(true, y_pred)

    print('#######################')
    print('ARI {}'.format(ARI))
    print('NMI {}'.format(NMI))

# Get hidden representation
hidden_model = Model(inputs=model.input, outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)
hidden = hidden.astype(float)

mid_str = dataset_str
hidden = pd.DataFrame(hidden)
hidden.to_csv('result/hidden_'+mid_str+'.tsv', sep='\t')

print('Done.')
