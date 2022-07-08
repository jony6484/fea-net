import torch
import matplotlib.pyplot as plt
from src.utils import read_data_file, make_dataloaders, make_tensors
from src.modeling import make_model
from src.training import train_model
import src.consts as consts




def main():
    data_df = read_data_file()
    data, labels, labels_mean, labels_std, data_mean, data_std = make_tensors(data_df)
    loader_train, loader_test = make_dataloaders(data, labels)
    net, loss_fun, optimizer = make_model(in_dim=data.shape[-1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss, test_loss, train_metric, test_metric, net = train_model(loader_train, loader_test, net, loss_fun,
                                                                        optimizer, device, num_epochs=500)
    print('eof')


if __name__ == "__main__":
    main()


