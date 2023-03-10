import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import r2_score
import wandb
import src.consts as consts
from torch.optim import lr_scheduler


def do_epoch(net, optimizer, scheduler, loss_fun, loader, device, train_mode=False):
    
    net.train(train_mode)
    batch_loss = []
    batch_metric = []
    for ii, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        
        if train_mode:
            y_hat = net(X)
            loss = loss_fun(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                y_hat = net(X)
                loss = loss_fun(y_hat, y)
        batch_loss.append(loss.item())
        batch_metric.append(100*r2_score(y.cpu(), y_hat.detach().cpu()))
    
    return np.mean(batch_loss), np.mean(batch_metric)
        




def train_model(train_loader, test_loader, net, loss_fun, optimizer, scheduler, device, training_params, model_params, num_epochs=2):
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
    conf = dict()
    conf.update(model_params)
    conf.update(training_params)
    wandb.config = conf
    wandb.watch(models=net, log='all')
    checkpoint_text = ""
    train_loss, test_loss, train_metric, test_metric = [], [], [], []
    best_model = {'test_metric': -1e16, 'epoch': -1, 'net': None}
    Nb = len(iter(train_loader))
    net.to(device)
    # Epochs:
    for epoch_i in range(num_epochs):
        # Train
        epoch_loss, epoch_metric = do_epoch(net, optimizer, scheduler, loss_fun, train_loader, device, train_mode=True)
        train_loss.append(epoch_loss)
        train_metric.append(epoch_metric)
        wandb.log({"train_loss": train_loss[-1]})
        wandb.log({"train_metric": train_metric[-1]})
        # Test
        epoch_loss, epoch_metric = do_epoch(net, optimizer, scheduler, loss_fun, test_loader, device, train_mode=False)
        test_loss.append(epoch_loss)
        test_metric.append(epoch_metric)
        wandb.log({"test_loss": test_loss[-1]})
        wandb.log({"test_metric": test_metric[-1]})
        # Save checkpoint for best model
        if test_metric[-1] > best_model['test_metric']:
            best_model['test_metric'] = test_metric[-1]
            best_model['epoch'] = epoch_i
            best_model['net'] = deepcopy(net.state_dict())
            checkpoint_text = " *checkpoint*"
        # Epoch print
        print(f'epoch:{epoch_i + 1}/{num_epochs}|loss Train/Test: {train_loss[epoch_i]:0.3e}/{test_loss[epoch_i]:0.3e}|'
              f'metric Train/Test: {train_metric[epoch_i]:0.3f}/{test_metric[epoch_i]:0.3f}%|'
              f'{checkpoint_text}')
        checkpoint_text = ""
    # Reloading and returning the model  
    net.load_state_dict(best_model['net'])
    net.to('cpu')
    metrics = {'train_loss': train_loss, 'test_loss': test_loss, 'train_metric': train_metric, 'test_metric': test_metric}
    return metrics, net
