import os
import zipfile
import numpy as np
import torch
import pandas as pd

def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)
    print(X.shape)
    
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds
    
def load_binghai_data():
    data_path = 'F:\农学时空预测-新-终版\data.npy'   # 修改
    data = np.load(data_path, allow_pickle=True).transpose((1, 2, 0))
 #   print(data.shape)
    A = np.load("F:\农学时空预测-新-终版\\adj.npy", allow_pickle=True).astype(np.float32)  #  修改
  #  X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = data.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds  

def load_binghai1_data():
    data_path = r'data1.npy'   # 修改
    data1 = np.load(data_path, allow_pickle=True).transpose((1, 2, 0))

    data = data1[:,0,:][:,np.newaxis,:]

 #   print(data.shape)
    A = np.load(r'adj.npy', allow_pickle=True).astype(np.float32)  #  修改
  #  X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = data.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    X = np.concatenate((X, data1[:,1:,:].astype(np.float32)), axis=1)

    return A, X, means, stds


def load_binghai2_data():
    data_path = r'data1.npy'  # 修改
    data1 = np.load(data_path, allow_pickle=True).transpose((1, 2, 0))

    sub_aera_name = [i for i in range(1, 12+1)]
    xy_data = []
    for sub_name in sub_aera_name:
        sub_data_path = f'region-disease\\region{sub_name}.csv'
        sub_file = np.array(pd.read_csv(sub_data_path, header=None))[:, 1:]

        # fixme 测试语句
        # sub_file = data1[sub_name-1].T
        # sub_file = sub_file[:, 1:]

        sub_file = sub_file.astype(np.float32)
        xy_data.append(sub_file.T)

    xy_data = np.stack(xy_data, axis=0)

    data = data1[:, 0, :][:, np.newaxis, :]

    data1 = xy_data

    #   print(data.shape)
    A = np.load(r'adj.npy', allow_pickle=True).astype(np.float32)  # 修改
    #  X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = data.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    X = np.concatenate((X, data1.astype(np.float32)), axis=1)

    return A, X, means, stds



#####扩展实验
# def load_binghai2_data():
#     data_path = r'data1.npy'  # 修改
#     data1 = np.load(data_path, allow_pickle=True).transpose((1, 2, 0))
#
#     sub_aera_name = [i for i in range(1, 12+1)]
#     xy_data = []
#     for sub_name in sub_aera_name:
#         sub_data_path = f'region-disease\\region{sub_name}.csv'
#         sub_file = np.array(pd.read_csv(sub_data_path, header=None))[:, 1:]
#
#         # fixme 测试语句
#         # sub_file = data1[sub_name-1].T
#         # sub_file = sub_file[:, 1:]
#
#         sub_file = sub_file.astype(np.float32)
#         xy_data.append(sub_file.T)
#
#     xy_data = np.stack(xy_data, axis=0)
#
#     data = data1[:, 0, :][:, np.newaxis, :]
#
#     data1 = xy_data
#
#     #   print(data.shape)
#     A = np.load(r'adj.npy', allow_pickle=True).astype(np.float32)  # 修改
#     #  X = np.load("data/node_values.npy").transpose((1, 2, 0))
#     X = data.astype(np.float32)
#
#     # Normalization using Z-score method
#     means = np.mean(X, axis=(0, 2))
#     X = X - means.reshape(1, -1, 1)
#     stds = np.std(X, axis=(0, 2))
#     X = X / stds.reshape(1, -1, 1)
#
#
#     X = np.concatenate((X, data1.astype(np.float32)), axis=1)
#
#     ####扩展实验：取前20%
#
#     # X= X[:, :, :int(1034 * 1)]  # 取出前 20% 的样本
#
#     return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def generate_dataset1(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, :, i + num_timesteps_input: j])


    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))