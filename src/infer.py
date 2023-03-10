import torch
import matplotlib.pyplot as plt
from utils import get_run, resample_p_h, set_random_state, load_config, read_data_file, make_dataloaders, make_tensors, resample_P_H_mats, prepare_X_Y
from modeling import FeaLstm, FeaCnn, make_model, get_last_run, save_model_and_params
import consts as consts
import torch.nn as nn
import numpy as np
import pandas as pd



def load_trained_model(checkpoint):
    config = checkpoint['config']
    model_type = config['training_params']['model']
    model_params = config[f'{model_type}_params']
    net = FeaLstm(in_dim=2, hidden_lstm_dim=model_params['lstm_hidden_dim'], hidden_fc_dim=model_params['fc_hidden_dim'], out_dim=3, bidi=model_params['bidi'], num_layers=model_params['num_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    X_norm = checkpoint['X_norm']
    Y_norm = checkpoint['Y_norm']    
    return net, config, X_norm, Y_norm


def make_inference(h, p, net, config, X_norm, Y_norm):
    seq_len = config['training_params']['sequence_len']
    h, p = resample_p_h(h, p, h_max=200, n_points=350)
    X = torch.tensor(np.stack([p, h], axis=0), dtype=torch.float32).T[None, :, :]
    X_mean = X_norm['mean'] 
    X_std = X_norm['std'] 
    X = (X - X_mean) / X_std
    
    Y_mean = Y_norm['mean'] 
    Y_std = Y_norm['std'] 
    
    Y_hat = net(X)    
    Y_hat = Y_hat * Y_std + Y_mean
    return Y_hat.detach().squeeze().numpy()


def main():
    checkpoint = get_run('lstm_general_run__14.pk')
    net, config, X_norm, Y_norm = load_trained_model(checkpoint)
    exp_file_name = 'data/new_data/exp_clean.csv'
    pd_xp = pd.read_csv(exp_file_name)
    pd_xp.columns = ['mm', 'gpa', 'atm']
    h = pd_xp['mm'].values
    p = pd_xp['gpa'].values
    Y_hat = make_inference(h, p, net, config, X_norm, Y_norm)
    

    
    
    print('end')

if __name__ == "__main__":
    main()