import matplotlib.pyplot as plt
from utils.rnn import MultitaskRNN, get_model, run_model
from utils.sequences import add_task_identity
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA


def plot_behavior(task_names, num_hidden, hparams, period_duration=50):
    
    # Retrieve the model and task information
    rnn, tasks = get_model(task_names, num_hidden, hparams)
    
    # Iterate over each task
    for task_index in range(len(tasks)):
        task_name = tasks[task_index].__class__.__name__
        inputs, outputs, output_pred, _ = run_model(rnn, tasks, task_index, period_duration)

        # Iterate over each input sequence
        for i in range(inputs.shape[0]):
            true_output = outputs[i].squeeze().detach().numpy()
            network_output = output_pred[i].squeeze().detach().numpy()

            # Plot true and network outputs in separate plots
            plt.figure()
            for j, (true_dim, network_dim) in enumerate(zip(true_output.T, network_output.T)):
                color = plt.cm.get_cmap('tab10')(j)  # Get a unique color for each output dimension
                plt.plot(true_dim, label='True Output', color=color)
                plt.plot(network_dim, label='Network Output', linestyle='--', color=color)
            
            plt.xlabel('Time Step')
            plt.ylabel('Output')
            plt.title(f'Behavior of Network on {task_name} - Sequence {i+1}')
            plt.legend()
            plt.show()

def plot_pca(data, feature_data, plot_feature_data=False):
    """
    Apply PCA on the data and plot the first two principal components. Optionally plot the feature data as well.

    Args:
        data (numpy.ndarray): The data to be visualized using PCA. This should be a 2D array with shape (n_samples, n_features).
        feature_data (numpy.ndarray): The feature data used to fit the PCA model. This should be a 2D array with shape (n_samples, n_features).
        plot_feature_data (bool, optional): Whether to also plot the feature data. Defaults to False.

    Returns: 
        None
    """
    # Fit the PCA model on the feature_data
    pca = PCA(n_components=2)
    pca.fit(feature_data)
    
    # Transform the data
    data_pca = pca.transform(data)
    
    # Plot the first two principal components of the data
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, label='Data')
    
    if plot_feature_data:
        # Transform the feature_data
        feature_data_pca = pca.transform(feature_data)
        
        # Plot the first two principal components of the feature_data
        plt.scatter(feature_data_pca[:, 0], feature_data_pca[:, 1], alpha=0.5, label='Feature data')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of data')
    plt.legend()
    plt.grid(True)
    plt.show()

