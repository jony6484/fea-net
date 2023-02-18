import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
from scipy.signal import resample
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm

def resample_p_h(h, p, h_max, n_points):
    h_last = h_max if h.max() >= h_max else h.max()
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
    return H_mat_resampl, P_mat_resampl
    

def make_H_P_C_mats(dataset):
    c_cols = [col for col in dataset.columns if col.startswith('C')]
    h_cols = [col for col in dataset.columns if col.startswith('h')]
    p_cols = [col for col in dataset.columns if col.startswith('p')]
    # (2, N, L) : N - numper of simulations, L - number of points
    H_mat = dataset[h_cols].values
    P_mat = dataset[p_cols].values
    N = dataset.shape[0]
    H_mat = np.concatenate([np.zeros((N, 1)), H_mat], axis=1)
    P_mat = np.concatenate([np.zeros((N, 1)), P_mat], axis=1)
    C_mat = dataset[c_cols].values
    return H_mat, P_mat, C_mat




def main():
    data_path = Path.cwd() / 'data' / 'new_data'
    data_filename = 'data_clean.csv'
    exp_file_name = 'exp_clean.csv'
    dataset = pd.read_csv(data_path / data_filename).iloc[:, 1:].copy()
    H_mat, P_mat, C_mat = make_H_P_C_mats(dataset)
    H_mat_resampl, P_mat_resampl = resample_P_H_mats(H_mat, P_mat, h_max=200, n_points=120)
    print('end')
    X = np.stack([H_mat_resampl, P_mat_resampl], axis=0)
    fig = px.line(x=X[0,0,:], y=X[1,0,:])
    for ii in range(1, X.shape[1], 4):
        fig.add_scatter(x=X[0,ii,:], y=X[1,ii,:])
    pd_xp = pd.read_csv('data/new_data/exp_clean.csv')
    pd_xp.columns = ['mm', 'gpa', 'atm']
    fig.add_scatter(x=pd_xp['mm'], y=pd_xp['gpa'])
    fig.data[-1].line.width = 5
    fig.show()
    
    return
    





if __name__ == "__main__":
    main()
    print('end`')
