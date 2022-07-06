import torch
import torch.nn as nn
import src.consts as consts


class FeaLstm(nn.Module):
    def __init__(self, in_dim, hidden_lstm_dim, hidden_fc_dim, out_dim):
        super(FeaLstm, self).__init__()
        self.hidden_lstm_dim = hidden_lstm_dim
        self.hidden_fc_dim = hidden_fc_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.lstm = nn.LSTM(self.in_dim, self.hidden_lstm_dim, batch_first=True)
        self.hidden_layer = nn.Linear(self.hidden_lstm_dim, self.hidden_fc_dim)
        self.out_layer = nn.Linear(self.hidden_fc_dim, 3)
        self.activation = nn.ReLU()

    def forward(self, X):
        out, (h_n, c_n) = self.lstm(X)
        output = self.activation(self.hidden_layer(h_n).squeeze())
        output = self.out_layer(output)
        return output


def make_model(in_dim):
    net = FeaLstm(in_dim=in_dim, hidden_lstm_dim=consts.HIDDEN_LSTM_DIM, hidden_fc_dim=consts.HIDDEN_FC_DIM, out_dim=3)
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=consts.LEARNING_RATE)
    return net, loss_fun, optimizer
