
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    elif dataset_name == 'md':
        n_vertex = 404
    elif dataset_name == 'md_cell':
        n_vertex = 404

    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'count.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    """시계열 데이터를 입력과 출력 형태로 변환합니다.

    Args:
        data: 이미 스케일링된 데이터
        n_his: 입력으로 사용할 과거 시점의 수
        n_pred: 예측할 미래 시점의 수
        device: 텐서를 저장할 디바이스
    """
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred + 1

    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head:tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device) 