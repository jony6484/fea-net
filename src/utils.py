import numpy as np
import pandas as pd
import src.consts as consts
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from scipy.fftpack import fft, ifft


def read_data_file():
    df = pd.read_excel(consts.DATA_PATH, header=None).T
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)
    df = df.astype('float32')
    return df


def make_tensors(df):
    cols = list(df.columns)
    labels_cols = [col for col in cols if col.startswith('C')]
    height_cols = [col for col in cols if col.startswith('height')]
    pressure_cols = [col for col in cols if col.startswith('pressure')]
    labels = torch.tensor(df[labels_cols].values)
    labels_mean = labels.mean(dim=0)
    labels_std = labels.std(dim=0)
    labels = (labels - labels_mean) / labels_std
    pressures = torch.tensor(df[pressure_cols].values)*1e6
    heights = torch.tensor(df[height_cols].values)
    heights_fft_amp = torch.tensor(np.abs(fft(heights.numpy())))
    heights_fft_ang = torch.tensor(np.angle(fft(heights.numpy())))
    # data = torch.concat([pressures[:, :, None], heights[:, :, None], heights_fft_amp[:, :, None], heights_fft_ang[:, :, None]], dim=2)
    data = torch.cat([pressures[:, :, None], heights[:, :, None]], dim=2)
    data_mean = data.mean(dim=(0,1))
    data_std = data.std(dim=(0,1))
    data = (data - data_mean) / data_std
    return data, labels, labels_mean, labels_std, data_mean, data_std


def make_dataloaders(data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                        test_size=consts.TEST_SIZE, shuffle=True)
    dataset_train = TensorDataset(data_train, labels_train)
    dataset_test = TensorDataset(data_test, labels_test)
    loader_train = DataLoader(dataset_train, batch_size=consts.BATCH_SIZE, shuffle=True, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=consts.BATCH_SIZE, shuffle=False, drop_last=False)
    return loader_train, loader_test



