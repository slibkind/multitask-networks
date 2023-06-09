import torch
from torch import optim

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

def find_attractors(model, inputs, initial_hidden, num_iterations, n_last):
    """
    Finds the attracting fixed points by running the model forward for a given number of iterations and returns the last n_last elements of the hidden trajectory.

    Args:
        model (nn.Module): The model to be run.
        inputs (torch.Tensor): The input tensor of shape (batch_size, n_inputs).
        initial_hidden (torch.Tensor): The initial hidden state tensor of shape (batch_size, n_hidden).
        num_iterations (int): Number of iterations to run the model forward.
        n_last (int): Number of last hidden states to return.

    Returns:
        torch.Tensor: The hidden state trajectory tensor of shape (batch_size, n_last, n_hidden).
    """
    batch_size = inputs.size(0)

    inputs = inputs.unsqueeze(1).expand(batch_size, num_iterations, inputs.size(1))
    _, hidden_trajectory = model(inputs, initial_hidden)

    return hidden_trajectory[:, -n_last:]

