import numpy as np
import pandas as pd
import consts as consts
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from scipy.fftpack import fft, ifft
from scipy.interpolate import griddata
from tqdm import tqdm
import yaml
import pickle
import random

def resample_p_h(h, p, h_max, n_points):
    h_last = h_max if max(h) >= h_max else max(h)
    new_h = np.sort(np.random.uniform(low=0.0, high=h_last, size=n_points))
    new_h[-1] = h_last
    new_h[0] = 0
    # new_p = griddata(h[new_inds], p[new_inds], new_h, method='cubic')
    new_p = griddata(h, p, new_h, method='cubic')
    return new_h, new_p


def resample_P_H_mats(H, P, h_max, n_points):
    N = H.shape[0]
    H_mat_resampl = np.zeros(shape=(N, n_points))
    P_mat_resampl = np.zeros(shape=(N, n_points))
    for ii in tqdm(range(N)):
        H_mat_resampl[ii], P_mat_resampl[ii] = resample_p_h(H[ii], P[ii], h_max, n_points=n_points)
        
    return torch.tensor(H_mat_resampl, dtype=torch.float32), torch.tensor(P_mat_resampl, dtype=torch.float32)
    

def read_data_file():
    # df = pd.read_excel(consts.DATA_PATH, header=None).T
    dataset = pd.read_csv(consts.DATA_PATH).iloc[:, 1:].copy()
    return dataset



def make_tensors(dataset):
    c_cols = [col for col in dataset.columns if col.startswith('C')]
    h_cols = [col for col in dataset.columns if col.startswith('h')]
    p_cols = [col for col in dataset.columns if col.startswith('p')]
    # (2, N, L) : N - numper of simulations, L - number of points
    H_mat = torch.tensor(dataset[h_cols].values, dtype=torch.float32)
    P_mat = torch.tensor(dataset[p_cols].values, dtype=torch.float32)
    N = dataset.shape[0]
    H_mat = torch.concatenate([torch.zeros((N, 1)), H_mat], axis=1)
    P_mat = torch.concatenate([torch.zeros((N, 1)), P_mat], axis=1)
    C_mat = torch.tensor(dataset[c_cols].values, dtype=torch.float32)
    return H_mat, P_mat, C_mat


def prepare_X_Y(H_mat, P_mat, C_mat):
    X = torch.cat([P_mat[:, :, None], H_mat[:, :, None]], dim=2)
    X_mean = X.mean(dim=(0,1))
    X_std =  X.std(dim=(0,1))
    X = (X - X_mean) / X_std
    Y_mean = C_mat.mean(dim=0)
    Y_std = C_mat.std(dim=0)
    Y = (C_mat - Y_mean) / Y_std
    return {'X': X, 'Y': Y, 'X_norm': {'mean': X_mean, 'std': X_std}, 'Y_norm': {'mean': Y_mean, 'std': Y_std}}
    

def make_dataloaders(data, labels, training_params):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                        test_size=training_params['test_size'], shuffle=True)
    dataset_train = TensorDataset(data_train, labels_train)
    dataset_test = TensorDataset(data_test, labels_test)
    loader_train = DataLoader(dataset_train, batch_size=training_params['batch_size'], shuffle=True, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=training_params['batch_size'], shuffle=False, drop_last=False)
    return loader_train, loader_test


def load_config():
    with consts.CONFIG_PATH.open(mode='r') as file:
        config = yaml.safe_load(file)
    return config



    
    
def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def get_run(file_name):
    with (consts.MODEL_PATH / file_name).open(mode='rb') as file:
        saved_data = pickle.load(file)
    return saved_data