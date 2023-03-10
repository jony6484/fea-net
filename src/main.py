import torch
import matplotlib.pyplot as plt
from src.utils import set_random_state, load_config, read_data_file, make_dataloaders, make_tensors, resample_P_H_mats, prepare_X_Y
from src.modeling import make_model, get_last_run, save_model_and_params
from src.training import train_model
import src.consts as consts
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR




def main():
    set_random_state(27)
    config = load_config()
    training_params = config['training_params']
    model_params = config[f"{training_params['model']}_params"]
    data_df = read_data_file()
    H_mat, P_mat, C_mat = make_tensors(data_df)
    seq_len=training_params['sequence_len']
    H_mat, P_mat = resample_P_H_mats(H_mat, P_mat, h_max=training_params['h_max'], n_points=seq_len)
    data_dict = prepare_X_Y(H_mat, P_mat, C_mat)
    X = data_dict['X']
    Y = data_dict['Y']
    X_norm = data_dict['X_norm']
    Y_norm = data_dict['Y_norm']
    
    loader_train, loader_test = make_dataloaders(X, Y, training_params)
    net, loss_fun, optimizer = make_model(in_dim=X.shape[-1], model_params=model_params, training_params=training_params)
    
    if training_params['continue_last_run']:
        saved_data = get_last_run()
        state_dict = saved_data['state_dict']
        config = saved_data['config']
        model_params = config[f"{training_params['model']}_params"]
        net.load_state_dict(state_dict)

    if training_params['schedueler']:
        scheduler    = OneCycleLR(optimizer, max_lr=1e-4, total_steps=training_params['num_epochs'] * len(loader_train))
    else:
        scheduler = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics, net = train_model(train_loader=loader_train, test_loader=loader_test, net=net, loss_fun=loss_fun,
                                                                        optimizer=optimizer, scheduler=scheduler, device=device, training_params=training_params,
                                                                        model_params=model_params, num_epochs=training_params['num_epochs'])
    save_model_and_params(net, config, metrics, X_norm, Y_norm)
    print('eof')


if __name__ == "__main__":
    main()


