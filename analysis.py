from utils.plotting import plot_behavior, plot_pca
from utils.sequences import add_task_identity
from utils.analysis import get_all_hiddens, minimize_speed, get_attractors
from utils.rnn import MultitaskRNN, get_model, run_model

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

task_names = ['DelayGo', 'DelayAnti']
num_hidden = 255
seed = 2
lr = -7
dt = 20
tau = 100

hparams = {
    'batch_size': 64, 
    'learning_rate': 10**(lr/2),
    'sigma_x': 0.1,
    'alpha': dt/tau,
    'sigma_rec': 0.05,
    'activation': 'softplus',
    'w_in_coeff': 1.0,
    'w_rec_coeff': 0.9, 
    'w_rec_init': 'diag',
    'l1_lambda': 0.0,
    'l2_lambda': 1e-7,
    'l1_h_lambda': 0.0,
    'l2_h_lambda': 1e-7,
    'seed': seed
}

rnn, tasks = get_model(task_names, num_hidden, hparams)

all_hiddens = get_all_hiddens(rnn, tasks)


# parameters for finding fixed points
lr = 0.1
q_thresh = 1e-4
r = 0.1   # proportion of all hidden states to be sampled

k = all_hiddens.size(0)
num_samples = int(k * r)  # Calculate the number of samples as 10% of k

indices = torch.randperm(k)[:num_samples]  # Randomly permute the indices and select the first num_samples
sampled_hiddens = all_hiddens[indices]  # Select the sampled hidden points using the sampled indices



period = "delay"
stimulus = 2
task_idx = 0

input = add_task_identity(tasks[task_idx].get_input(period, stimulus), task_idx, len(tasks))

fps = minimize_speed(rnn, input, sampled_hiddens, lr, q_thresh)

plot_pca(fps.detach().numpy(), all_hiddens.detach().numpy(), plot_feature_data=True)  # convert tensors to numpy arrays for use with sklearn