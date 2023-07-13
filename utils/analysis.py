import torch
from torch import optim

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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

    
def minimize_speed(model, input, initial_hidden, learning_rate, q_thresh, max_iterations=None, verbose=True, method='first', grad_threshold=1e-4, check_interval=10000):
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
        max_iterations (int, optional): The maximum number of iterations to run the optimization. If None, the optimization runs indefinitely until the condition is met.
        grad_threshold (float, optional): The threshold for the gradient norm at which we consider a hidden state to have reached a local minimum. Defaults to 1e-4.
        check_interval (int, optional): The number of iterations between checks for local minima. Defaults to 10000.

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
    start_time = time.time()

    while True:
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

        # Every check_interval iterations, check if any hidden states have reached a local minimum with speed above the threshold
        if iteration % check_interval == 0:
            # Get the current gradients
            grad_norms = torch.norm(hidden.grad.data, dim=1)

            # Find which hidden states are in a local minimum (i.e., their gradient norm is below the threshold)
            local_minima = grad_norms < grad_threshold

            # If any hidden states are in a local minimum, check their speed
            if torch.any(local_minima):
                speed_per_init_cond = get_speed(model, input, hidden.detach())
                # Find which of these hidden states have speed above the threshold
                bad_local_minima = local_minima & (speed_per_init_cond > q_thresh)

                # If any hidden states are in a bad local minimum, remove them
                if torch.any(bad_local_minima):
                    if verbose: 
                        num_bad_minima = bad_local_minima.sum().item()
                        print(f"Removing {num_bad_minima} bad local minima at iteration {iteration}.")
                    good_indices = ~bad_local_minima
                    hidden = hidden[good_indices].clone().detach().requires_grad_(True)
                    inputs = input.repeat(hidden.shape[0], 1)
        
        if verbose and iteration % check_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}: maximum speed across all initial conditions is {torch.max(speed_per_init_cond).item()}")
            print(f"Time taken for the last 10000 iterations: {elapsed_time} seconds.")
            start_time = time.time()

        iteration += 1

        if torch.max(speed_per_init_cond).item() < q_thresh: 
            if verbose:
                print("Stopping optimization: maximum speed across all initial conditions is below the threshold.")
            break

        if max_iterations is not None and iteration >= max_iterations:
            if verbose:
                print(f"Stopping optimization: reached maximum number of iterations ({max_iterations}).")
            break

    return hidden

def get_unique_fixed_points(fixed_points):
    """
    Apply DBSCAN clustering algorithm on given fixed points to find unique fixed points.

    The function takes as input a torch tensor of fixed points and applies the DBSCAN algorithm to cluster these points. 
    After clustering, the mean of each cluster is calculated to serve as the representative point for that cluster.
    This function is designed to reduce the redundancy of fixed points obtained from some process.

    Args:
        fixed_points (torch.Tensor): Input tensor containing fixed points.

    Returns:
        torch.Tensor: A tensor containing the representative fixed points.
    """

    # Convert to numpy for compatibility with DBSCAN
    fixed_points_np = fixed_points.numpy()

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=1)  # You may want to adjust these parameters
    dbscan.fit(fixed_points_np)

    # Group fixed points into clusters and calculate eigenvalues for each unique point
    unique_labels = set(dbscan.labels_)

    # Preallocate representatives as a numpy array
    representatives = np.empty((len(unique_labels), fixed_points_np.shape[1]))

    i = 0
    for label in unique_labels:
        idx = dbscan.labels_ == label
        cluster_points = fixed_points_np[idx]
        # Find a representative point for this cluster
        representative_point = cluster_points.mean(axis=0)
        representatives[i] = representative_point
        i += 1

    return torch.from_numpy(representatives)




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

