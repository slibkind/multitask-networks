import torch
from torch import optim

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.model import run_model



def get_all_hiddens(rnn, tasks):
    """
    Runs the given RNN model on all tasks and returns the concatenated hidden states.

    Args:
        rnn (nn.Module): The RNN model.
        tasks (list): List of tasks.

    Returns:
        torch.Tensor: The concatenated hidden states of shape (n_sequences * time_steps, n_hidden).
    """
    all_hiddens = []
    for task_index in range(len(tasks)):
        _, _, _, hiddens = run_model(rnn, tasks, task_index)
        all_hiddens.append(hiddens)

    # Concatenate the hidden states from all tasks
    concatenated_hiddens = torch.cat(all_hiddens, dim=0)

    # Reshape the concatenated hidden states
    reshaped_hiddens = concatenated_hiddens.view(-1, concatenated_hiddens.size(-1))

    return reshaped_hiddens


def get_attractors(model, input, initial_hidden_states, num_timesteps, num_last):
    """
    Run the RNN forward for a given number of timesteps and return the output and hidden states trajectories.

    Args:
        model (torch.nn.Module): The RNN model.
        input (torch.Tensor): The input to the RNN. This should be a 1D tensor of shape (n_inputs,).
        initial_hidden_states (torch.Tensor): The initial hidden states for the RNN. This should be a 2D tensor of shape (batch_size, n_hidden).
        num_timesteps (int): The number of timesteps to run the RNN forward.
        num_last (int): The number of hidden states from the end to return.

    Returns: 
        torch.Tensor, torch.Tensor: The output trajectory and hidden state trajectory. Both are 3D tensors of shape (batch_size, timesteps, feature_size).
    """
    # Expand the input to the correct shape
    expanded_input = input.unsqueeze(0).unsqueeze(1).expand(initial_hidden_states.size(0), num_timesteps, -1)

    # Run the RNN forward
    _, hidden_state_trajectory = model.forward_trajectory(expanded_input, initial_hidden_states)
    
    return hidden_state_trajectory[:, -num_last:]


def get_speed(model, input, hidden, duplicate_input=True):
    """
    Computes the speed (q) of the dynamics for a given model and a set of hidden states. 

    The speed is computed for each hidden state. Depending on the value of `duplicate_input`, the function
    either duplicates the input for all hidden states, or uses the input as is.

    Args:
        model (nn.Module): The multitask RNN model.
        input (torch.Tensor): The input sequence. If the shape is (num_inputs), this input will be used 
            for all hidden states. If the shape is (num_hidden_states, num_inputs), each hidden state will 
            use the corresponding row from this tensor as its input.
        hidden (torch.Tensor): A tensor containing a set of hidden states for which to compute the speed. 
            The shape should be (num_hidden_states, num_hidden). Each row corresponds to a different hidden state.
        duplicate_input (bool, optional): Whether to duplicate the input for all hidden states. If True, the input
            will be duplicated regardless of its shape. If False, the input will be used as is. Defaults to True.

    Returns:
        torch.Tensor: A tensor of shape (num_hidden_states,) containing the computed speed for each hidden state.
    """

    if duplicate_input:
        inputs = input.repeat(hidden.shape[0], 1)
    else:
        inputs = input

    # Forward pass to compute the dynamics
    dhidden = -hidden + model.activation(torch.einsum('ij,bj->bi', model.W_in, inputs) + torch.einsum('ij,bj->bi', model.W_rec, hidden) + model.b)

    # Compute the speed for each hidden state
    speed = 0.5 * torch.norm(dhidden, dim=1) ** 2
    
    return speed

    
import time
import torch
from torch import optim

def minimize_speed(model, input, initial_hidden, learning_rate, q_thresh, verbose=True, method='first'):
    """
    Minimizes the speed (q) of the dynamics of a given model using gradient descent. The optimization
    is performed from multiple initial conditions simultaneously, which allows for more comprehensive 
    exploration of the hidden state space.

    Args:
        model (nn.Module): The multitask RNN model.
        input (torch.Tensor): The input sequence of shape (num_inputs). This single sequence is used for all initial conditions.
        initial_hidden (torch.Tensor): The initial hidden states of shape (num_initial_conditions, num_hidden).
            Each row corresponds to the initial hidden state for a different initial condition.
        learning_rate (float): Learning rate for gradient descent.
        q_thresh (float): A threshold for the speed. The optimization stops if the maximum speed across all initial conditions is less than this threshold.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        method (str, optional): Whether to use 'first' or 'second' order optimization. Defaults to 'first'.

    Returns:
        torch.Tensor: The updated hidden states that minimize the speed. Each row corresponds to the optimized 
        hidden state for a different initial condition.
    """

    # Check if a GPU is available and if not, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model and tensors to the chosen device
    model = model.to(device)
    input = input.to(device)
    initial_hidden = initial_hidden.to(device)

    # Create a tensor of inputs by duplicating the single input across all initial conditions
    inputs = input.repeat(initial_hidden.shape[0], 1)

    hidden = initial_hidden.detach().clone().requires_grad_(True)
    
    if method.lower() == 'first':
        optimizer = optim.SGD([hidden], lr=learning_rate)
    elif method.lower() == 'second':
        optimizer = optim.LBFGS([hidden], lr=learning_rate)
    else:
        raise ValueError("Unknown method: {}. Use 'first' or 'second'.".format(method))

    iteration = 0
    while True:
        start_time = time.time()
        speed_per_init_cond = None
        
        def closure():
            nonlocal speed_per_init_cond
            optimizer.zero_grad()

            # Forward pass to compute the dynamics
            dhidden = -hidden + model.activation(torch.einsum('ij,bj->bi', model.W_in, inputs) + torch.einsum('ij,bj->bi', model.W_rec, hidden) + model.b)

            # Compute the speed for each initial condition
            speed_per_init_cond = 0.5 * torch.norm(dhidden, dim=1) ** 2

            # Sum the speeds across all initial conditions to get the total speed
            total_speed = torch.sum(speed_per_init_cond)

            # Backward pass to compute the gradients
            total_speed.backward()
            
            return total_speed

        # Depending on the method, the optimizer step is different
        if method.lower() == 'first':
            closure()
            optimizer.step()
        elif method.lower() == 'second':
            optimizer.step(closure)
        
        if verbose and iteration % 10000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}: maximum speed across all initial conditions is {torch.max(speed_per_init_cond).item()}")
            print(f"Time taken for the last 10000 iterations: {elapsed_time} seconds.")

        iteration += 1

        if torch.max(speed_per_init_cond).item() < q_thresh: 
            if verbose:
                print("Stopping optimization: maximum speed across all initial conditions is below the threshold.")
            break

    return hidden




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
