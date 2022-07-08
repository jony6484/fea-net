import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import r2_score
import wandb
import src.consts as consts

def train_model(train_loader, test_loader, net, loss_fun, optimizer, device, num_epochs=2):
    """
    A function to train the model
    :param train_loader: train_loader
    :param test_loader: test_loader
    :param net: model
    :param loss_fun: loss_fun
    :param optimizer: optimizer
    :param device: device
    :param num_epochs: num_epochs
    :return: the training states and the model at its best state
    """
    # Setting up logging
    wandb.init(project=consts.PROJECT_NAME)
    wandb.config = {"learning_rate": consts.LEARNING_RATE,
                    "epochs": num_epochs,
                    "batch_size": consts.BATCH_SIZE}
    wandb.watch(models=net, log='all')
    checkpoint_text = ""
    train_loss, test_loss, train_metric, test_metric = [], [], [], []
    best_model = {'test_metric': -1e16, 'epoch': -1, 'net': None}
    Nb = len(iter(train_loader))
    net.to(device)
    for epoch_i in range(num_epochs):
        net.train()
        batch_loss = []
        batch_metric = []
        for ii, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = loss_fun(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.detach().item())
            batch_metric.append(100*r2_score(y.cpu(), y_hat.detach().cpu()))
            print(f'batch:{ii + 1}/{Nb}|{int(100 * ii / Nb) * "="}{int(100 * (Nb - ii) / Nb) * "-"}'
                  f'|loss:{batch_loss[ii]:0.3e}|metric:{batch_metric[ii]:0.3f}', end='\r')
        train_loss.append(np.mean(batch_loss))
        train_metric.append(np.mean(batch_metric))
        wandb.log({"train_loss": train_loss[-1]})
        wandb.log({"train_metric": train_metric[-1]})
        net.eval()
        with torch.no_grad():
            batch_loss = []
            batch_metric = []
            for ii, (X, y) in enumerate(test_loader):
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                loss = loss_fun(y_hat, y)
                batch_loss.append(loss.detach().item())
                batch_metric.append(100*r2_score(y.cpu(), y_hat.detach().cpu()))
            test_loss.append(np.mean(batch_loss))
            test_metric.append(np.mean(batch_metric))
            wandb.log({"test_loss": test_loss[-1]})
            wandb.log({"test_metric": test_metric[-1]})
        if test_metric[-1] > best_model['test_metric']:
            best_model['test_metric'] = test_metric[-1]
            best_model['epoch'] = epoch_i
            best_model['net'] = deepcopy(net.state_dict())
            checkpoint_text = " *checkpoint*"
        print(f'epoch:{epoch_i + 1}/{num_epochs}|loss Train/Test: {train_loss[epoch_i]:0.3e}/{test_loss[epoch_i]:0.3e}|'
              f'metric Train/Test: {train_metric[epoch_i]:0.3f}/{test_metric[epoch_i]:0.3f}%|'
              f'{checkpoint_text}')
        checkpoint_text = ""
    net.load_state_dict(best_model['net'])
    net.to('cpu')
    return train_loss, test_loss, train_metric, test_metric, net
