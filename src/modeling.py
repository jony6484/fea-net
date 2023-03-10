import torch
import torch.nn as nn
import consts as consts
import torch.nn.functional as F
import numpy as np
import pickle

class FeaCnn(nn.Module):
    def __init__(self, in_channels, conv_1_out_channels, conv_2_out_channels, conv_3_out_channels, seq_len, out_dim):
        super(FeaCnn, self).__init__()
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_1_out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(num_features=conv_1_out_channels)
        
        self.conv2 = nn.Conv1d(in_channels=conv_1_out_channels, out_channels=conv_2_out_channels, kernel_size=3, padding=1)
        self.bnorm2 = nn.BatchNorm1d(num_features=conv_2_out_channels)
        
        self.conv3 = nn.Conv1d(in_channels=conv_2_out_channels, out_channels=conv_3_out_channels, kernel_size=3, padding=1)
        self.bnorm3 = nn.BatchNorm1d(num_features=conv_3_out_channels)
        
        self.fc1_in_features = int(seq_len * conv_3_out_channels // 8)
        
        self.fc1 = nn.Linear(in_features=self.fc1_in_features, out_features=int(self.fc1_in_features // 2))
        self.out_layer = nn.Linear(in_features=int(self.fc1_in_features // 2), out_features=out_dim)
        # self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()
        

    def forward(self, X):
        L = X.shape[-1]
        X = F.max_pool1d(self.conv1(X.permute(0,2,1)), 2)
        X = self.activation(self.bnorm1(X))
        X = F.max_pool1d(self.conv2(X), 2)
        X = self.activation(self.bnorm2(X))
        X = F.max_pool1d(self.conv3(X), 2)
        X = self.activation(self.bnorm3(X))
        X = X.flatten(start_dim=1)
        X = self.activation(self.fc1(X))
        X = F.dropout(X, p=.5,training=self.training)
        X = self.out_layer(X)
        return X


class FeaLstm(nn.Module):
    def __init__(self, in_dim, hidden_lstm_dim, hidden_fc_dim, out_dim, bidi, num_layers):
        super(FeaLstm, self).__init__()
        self.hidden_lstm_dim = hidden_lstm_dim
        self.hidden_fc_dim = hidden_fc_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.bidi = bidi
        linear_input_dim = hidden_lstm_dim * (1 + bidi) * num_layers
        self.lstm = nn.LSTM(self.in_dim, self.hidden_lstm_dim, batch_first=True, bidirectional=bidi, num_layers=num_layers)
        self.hidden_layer = nn.Linear(linear_input_dim, self.hidden_fc_dim)
        self.out_layer = nn.Linear(self.hidden_fc_dim, 3)
        self.activation = nn.ReLU()

    def forward(self, X):
        out, (h_n, c_n) = self.lstm(X)
        h_n = h_n.permute(1,0,-1).flatten(start_dim=1)        
        X = self.activation(self.hidden_layer(h_n))
        X = F.dropout(X, p=.3,training=self.training)
        X = self.out_layer(X)
        return X


def make_model(in_dim, model_params, training_params):
    if training_params['model'] == 'lstm':
        net = FeaLstm(in_dim=in_dim, hidden_lstm_dim=model_params['lstm_hidden_dim'], hidden_fc_dim=model_params['fc_hidden_dim'], out_dim=3, bidi=model_params['bidi'], num_layers=model_params['num_layers'])
    else:
        net = FeaCnn(in_channels=in_dim, conv_1_out_channels=model_params['conv_1_out_chs'], conv_2_out_channels=model_params['conv_2_out_chs'],
                     conv_3_out_channels= model_params['conv_3_out_chs'], seq_len=training_params['sequence_len'], out_dim=3)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=training_params['lr'])
    return net, loss_fun, optimizer


def get_last_run():
    file_names = [f.name.split('.')[0] for f in consts.MODEL_PATH.glob('*.pk')]
    serial_nums = [int(f.split('__')[-1]) for f in file_names]
    file_name = file_names[np.argmax(serial_nums)] + ".pk"
    with (consts.MODEL_PATH / file_name).open(mode='rb') as file:
        saved_data = pickle.load(file)
    return saved_data

def save_model_and_params(net, config, metrics, X_norm, Y_norm):
    file_names = [f.name.split('.')[0] for f in consts.MODEL_PATH.glob('*.pk')]
    if len(file_names) > 0:
        serial_num = max([int(f.split('__')[-1]) for f in file_names]) + 1
    else:
        serial_num = 0
    file_name = f"{config['training_params']['model']}_general_run__{serial_num}.pk"
    with (consts.MODEL_PATH / file_name).open(mode='wb') as file:
        pickle.dump({'state_dict': net.state_dict(), 'config': config, 'metrics': metrics, 'X_norm': X_norm, 'Y_norm': Y_norm}, file)
    print(f"model saved as {file_name}")