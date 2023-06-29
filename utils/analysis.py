import torch
from torch import optim
from utils.rnn import run_model 

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



def minimize_speed(model, inputs, initial_hidden, learning_rate, num_iterations, q_thresh, verbose=True):
    """
    Minimizes the speed (q) of the dynamics of a given model using gradient descent.

    Args:
        model (nn.Module): The multitask RNN model.
        inputs (torch.Tensor): The input sequences of shape (batch_size, num_inputs).
        initial_hidden (torch.Tensor): The initial hidden state of shape (batch_size, num_hidden).
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        verbose (bool, optional): Whether to print the speed value. Defaults to True.

    Returns:
        torch.Tensor: The updated hidden state that minimizes the speed.
    """

    hidden = initial_hidden.detach().clone().requires_grad_(True)
    optimizer = optim.SGD([hidden], lr=learning_rate)    

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass to compute the dynamics
        dhidden = -hidden + model.activation(torch.einsum('ij,bj->bi', model.W_in, inputs) + torch.einsum('ij,bj->bi', model.W_rec, hidden) + model.b)

        # Compute the speed for each batch and sum the results
        speed = 0.5 * torch.sum(torch.norm(dhidden, dim=1) ** 2)

        if torch.max(dhidden).item() < q_thresh: 
            break

        # Backward pass to compute the gradients
        speed.backward()

        # Update the hidden state using gradient descent
        optimizer.step()

        if verbose and i % 1000 == 0:
            print(f"Speed at iteration {i}: {speed.item()}")
            print(f"Max speed: {torch.max(dhidden).item()}")

    return hidden