import torch
from torch import optim
from torch.autograd.functional import jacobian

import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from utils.model import run_model
from utils.task import get_input
from utils.utils import get_fixed_point_path, get_model

from sklearn.decomposition import PCA

def get_hidden_trajectories(rnn, tasks):
    """
    Runs the given RNN model on all tasks and returns the concatenated hidden trajectories.

    Args:
        rnn (nn.Module): The RNN model.
        tasks (list): List of tasks.

    Returns:
        torch.Tensor: The concatenated hidden trajectories of shape (n_sequences, timesteps, n_hidden).
    """
    all_hidden_trajectories = []
    for task_index in range(len(tasks)):
        _, _, _, hidden_trajectory = run_model(rnn, tasks, task_index)
        all_hidden_trajectories.append(hidden_trajectory)

    # Concatenate the hidden trajectories from all tasks
    concatenated_hidden_trajectories = torch.cat(all_hidden_trajectories, dim=0)

    return concatenated_hidden_trajectories

def get_all_hiddens(rnn, tasks):
    """
    Runs the given RNN model on all tasks and returns the concatenated hidden states.

    Args:
        rnn (nn.Module): The RNN model.
        tasks (list): List of tasks.

    Returns:
        torch.Tensor: The concatenated hidden states of shape (n_sequences * time_steps, n_hidden).
    """
    hidden_trajectories = get_hidden_trajectories(rnn, tasks)

    # Reshape the concatenated hidden states
    reshaped_hiddens = hidden_trajectories.view(-1, hidden_trajectories.size(-1))

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

        # Check if all local minima were "bad" and all have been removed
        if hidden.numel() == 0:
            if verbose:
                print("Stopping optimization: all local minima were 'bad' and have been removed.")
            break

        if torch.max(speed_per_init_cond).item() < q_thresh: 
            if verbose:
                print("Stopping optimization: maximum speed across all initial conditions is below the threshold.")
            break

        if max_iterations is not None and iteration >= max_iterations:
            if verbose:
                print(f"Stopping optimization: reached maximum number of iterations ({max_iterations}).")
            break

    return hidden

def get_fixed_points(model_name, input, q_thresh = None):
    """
    Load fixed points from a file and optionally filter them based on their speeds.

    This function loads fixed points for a given model and input from a file. If a speed threshold `q_thresh` is set,
    it filters the fixed points to only include those with speeds below the threshold.

    Args:
        model_name (str): The name of the model for which to load fixed points.
        model (nn.Module): The actual model instance used for computing speeds if `q_thresh` is provided.
        input (torch.Tensor): The input sequence. Used for computing speeds if `q_thresh` is provided.
        q_thresh (float, optional): A speed threshold. If set, only fixed points with speeds below this threshold 
            are returned. Defaults to None, in which case all fixed points are returned.

    Returns:
        torch.Tensor: A tensor containing the loaded fixed points. If `q_thresh` is set, this tensor only includes 
            fixed points with speeds below the threshold.
            
    Raises:
        FileNotFoundError: If no fixed point file is found for the given model and input.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_point_path = get_fixed_point_path(model_name, input)

    # Check if file exists
    if not os.path.exists(fixed_point_path):
        raise FileNotFoundError(f"No fixed point file found for input {input} and model {model_name}.")

    fixed_points = torch.load(fixed_point_path, map_location=device)
    
    # If q_thresh is set, filter the fixed points based on their speeds
    if q_thresh is not None:
        rnn, _ = get_model(model_name)
        speeds = get_speed(rnn, input, fixed_points)
        fixed_points = fixed_points[speeds < q_thresh]

    return fixed_points

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

    return torch.from_numpy(representatives).float()


def compute_jacobian(rnn, hidden_state, input):
    """Compute the Jacobian of the RNN's hidden state with respect to the input.
    Args:
        rnn (MultitaskRNN): The RNN model.
        hidden_state (torch.Tensor): The hidden state of the RNN.
        input (torch.Tensor): The input to the RNN.

    Returns:
        torch.Tensor: The Jacobian matrix.
    """

    def get_dx(x):
        # Make sure the tensor has gradient computation
        x.requires_grad_(True)

        # Update the hidden state with the RNN's state transition function
        dx = rnn.activation(
            torch.einsum('ij,j->i', rnn.W_in, input) + torch.einsum('ij,j->i', rnn.W_rec, x) + rnn.b)
        
        return dx

    # Compute the Jacobian using automatic differentiation
    return jacobian(get_dx, hidden_state)


def is_stable(rnn, fixed_point, input):
    """
    Determines the stability of a fixed point of a recurrent neural network (RNN) given an input. 

    A fixed point is considered stable if the real parts of all the eigenvalues of the Jacobian matrix at the fixed point are less than 1.

    Args:
        rnn (torch.nn.Module): The RNN model.
        fixed_point (torch.Tensor): The fixed point to evaluate the stability of. This tensor represents the state of the RNN.
        input (torch.Tensor): The input provided to the RNN.

    Returns:
        torch.Tensor: A boolean tensor indicating the stability of the fixed point. Returns True if the fixed point is stable, False otherwise.
    """

    # Compute the Jacobian matrix at the given fixed point
    jacobian = compute_jacobian(rnn, fixed_point, input)

    # Compute the eigenvalues of the Jacobian
    eigenvalues = torch.linalg.eig(jacobian).eigenvalues

    # Extract the real parts of the eigenvalues
    real_parts = eigenvalues.real

    # Check the stability condition
    return torch.all(real_parts < 1)




def plot_hiddens_and_data(rnn, tasks, data_list, label_list=None, color_list=None, filled_list=None, title=None):
    """
    Apply PCA on the hidden trajectories and plot the first two principal components of the data and hidden trajectories.

    Args:
        rnn (nn.Module): The RNN model.
        tasks (list): List of tasks.
        data_list (list of torch.Tensor): List of data tensors to be plotted.
        label_list (list of str, optional): List of labels corresponding to the data tensors. Defaults to None.
        color_list (list of str, optional): List of colors for the data tensors. Defaults to None.
        filled_list (list of bool, optional): List of booleans indicating whether each data tensor should be plotted as filled or open circles. Defaults to filled circles.

    Returns: 
        None
    """
    # Initialize default labels, colors and filled status if not provided
    if label_list is None:
        label_list = ['Data {}'.format(i) for i in range(len(data_list))]
    if color_list is None:
        color_list = ['k' for i in range(len(data_list))]
    if filled_list is None:
        filled_list = [True for _ in range(len(data_list))]

    # Get the hidden trajectories for all tasks
    all_hidden_trajectories = get_hidden_trajectories(rnn, tasks)
    
    # Reshape the all_hidden_trajectories tensor for PCA
    reshaped_all_hidden_trajectories = all_hidden_trajectories.view(-1, all_hidden_trajectories.size(-1))
    
    # Fit the PCA model on the reshaped_all_hidden_trajectories
    pca = PCA(n_components=2)
    pca.fit(reshaped_all_hidden_trajectories.detach().numpy())
    
    # Transform and plot the data tensors
    for data, label, color, filled in zip(data_list, label_list, color_list, filled_list):
        data_pca = pca.transform(data.detach().numpy())
        plt.scatter(data_pca[:, 0], data_pca[:, 1], label=label, 
                    facecolors=color if filled else 'none', 
                    edgecolors=color if not filled else 'none')

    # Create a color map that goes from blue to red
    colors = plt.cm.viridis(np.linspace(0, 1, all_hidden_trajectories.size(1)))  # number of timesteps

    # Iterate over the hidden trajectories and plot the PCA of each one
    for hidden_trajectory in all_hidden_trajectories:
        
        # Transform the reshaped hidden_trajectory
        hidden_trajectory_pca = pca.transform(hidden_trajectory.detach().numpy())

        # Plot the first two principal components of the reshaped_hidden_trajectory for each timestep
        for timestep in range(hidden_trajectory.size(0)):
            plt.scatter(hidden_trajectory_pca[timestep, 0], 
                        hidden_trajectory_pca[timestep, 1], 
                        alpha=0.25, 
                        s=10,
                        c=[colors[timestep]])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()




def visualize_fixed_points(model_name, task_idx, period, stimulus, n_interp, 
                           q_thresh=None,
                           input_labels=None, 
                           title=None, 
                           figsize=(10, 8), 
                           cmap='plasma', 
                           save_fig=None,
                           s = 100):
    """
    Visualize fixed points' first principal component for each interpolated input.
    
    Args:
        model_name (str): Name of the model used to generate the fixed points.
        task_idx (list of int): List of task indices.
        period (list of str): List of periods.
        stimulus (list of int): List of stimuli.
        n_interp (int): Number of interpolation steps.

    Returns:
        None
    """
    # Load model and tasks
    rnn, tasks = get_model(model_name)

    # Initialize color map
    cmap = plt.get_cmap(cmap)
    color_norm = cm.colors.Normalize(vmin=0, vmax=(len(task_idx)-1)*n_interp)  # Normalize to the range of interpolation steps across all pairs

    plt.figure(figsize=figsize)

    s = s   # set the size of the plotted points

    # List to store all fixed points
    all_fixed_points = []

    # Loop over the task pairs
    for t in range(len(task_idx) - 1):

        # Generate the task inputs
        input1 = get_input(task_idx[t], period[t], stimulus[t], tasks)
        input2 = get_input(task_idx[t+1], period[t+1], stimulus[t+1], tasks)

        # Get fixed points for each interpolated input
        for i in range(n_interp + 1):
            # Linearly interpolate between the inputs
            interpolated_input = (n_interp - i) / n_interp * input1 + i / n_interp * input2

            # Load the fixed points for the interpolated input
            fixed_points = get_fixed_points(model_name, interpolated_input, q_thresh=q_thresh)

            # Get unique fixed points
            unique_fixed_points = get_unique_fixed_points(fixed_points.detach())

            # Add to list
            all_fixed_points.append(unique_fixed_points)

    # Concatenate all fixed points to fit PCA
    concat_fixed_points = torch.cat(all_fixed_points, dim=0)

    # Convert to numpy for compatibility with PCA
    concat_fixed_points_np = concat_fixed_points.detach().numpy()

    # Apply PCA
    pca = PCA(n_components=1)  # We're only interested in the first principal component
    pca.fit(concat_fixed_points_np)

    # Plot the first principal component for each interpolated input
    for t in range(len(task_idx) - 1):
        
        # Generate the task inputs
        input1 = get_input(task_idx[t], period[t], stimulus[t], tasks)
        input2 = get_input(task_idx[t+1], period[t+1], stimulus[t+1], tasks)

        for i in range(n_interp + 1):
            # Linearly interpolate between the inputs
            interpolated_input = (n_interp - i) / n_interp * input1 + i / n_interp * input2
            
            fixed_points_i = all_fixed_points[t*n_interp+i]
            projections = pca.transform(fixed_points_i.detach().numpy())[:, 0]  # Only consider the first PC

            # Check if each fixed point is stable
            for fp, proj in zip(fixed_points_i, projections):
                stable = is_stable(rnn, fp, interpolated_input)
                if stable:
                    plt.scatter([(t + i / n_interp)], [proj], color=cmap(color_norm(t*n_interp+i)), edgecolors='none', s=s)
                else:
                    plt.scatter([(t + i / n_interp)], [proj], color='none', edgecolor=cmap(color_norm(t*n_interp+i)), s=s)
    
    # Draw a vertical line at each integer on the x-axis and optionally add a label
    for t in range(len(task_idx)):
        plt.axvline(x=t, linestyle='--', color='gray')

    # Set the xticks to be at the integers, and use your labels
    if input_labels:
        plt.xticks(range(len(task_idx)), input_labels, rotation=45)
            
    plt.xlabel('Interpolation Step')
    plt.ylabel('First Principal Component')

    if title:
        plt.title(title)

    if save_fig:
        plt.savefig(save_fig)

    else:
        plt.show()
