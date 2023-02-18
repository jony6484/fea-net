import torch
import matplotlib.pyplot as plt
from src.utils import read_data_file, make_dataloaders, make_tensors, resample_P_H_mats, prepare_X_Y
from src.modeling import make_model
from src.training import train_model
import src.consts as consts




def main():
    data_df = read_data_file()
    H_mat, P_mat, C_mat = make_tensors(data_df)
    H_mat, P_mat = resample_P_H_mats(H_mat, P_mat, h_max=200, n_points=120)
    data_dict = prepare_X_Y(H_mat, P_mat, C_mat)
    X = data_dict['X']
    Y = data_dict['Y']
    X_norm = data_dict['X_norm']
    Y_norm = data_dict['Y_norm']
    
    # data, labels, labels_mean, labels_std, data_mean, data_std = make_tensors(data_df)
    loader_train, loader_test = make_dataloaders(data, labels)
    net, loss_fun, optimizer = make_model(in_dim=data.shape[-1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss, test_loss, train_metric, test_metric, net = train_model(loader_train, loader_test, net, loss_fun,
                                                                        optimizer, device, num_epochs=500)
    print('eof')


if __name__ == "__main__":
    main()


