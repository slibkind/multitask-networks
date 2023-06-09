import torch
import torch.nn.functional as F

def smooth_sequence(sequence, window):
    """
    Smooths a sequence tensor along a specified axis using convolution with a diagonal kernel.

    Args:
        sequence (torch.Tensor): The input sequence tensor of shape (batch_size, input_length, num_channels).
        window (int): The size of the smoothing window.

    Returns:
        torch.Tensor: The smoothed sequence tensor of the same shape as the input.

    """

    # Create a diagonal kernel tensor with entries 1/window
    kernel = torch.zeros(sequence.shape[-1], sequence.shape[-1], window)
    kernel.diagonal(dim1=0, dim2=1).fill_(1 / window)

    # Determine the padding size based on the window size
    padding = (window - 1) // 2 if window % 2 == 1 else window // 2

    # Perform convolution along the specified axis
    smoothed_sequence = F.conv1d(sequence.permute(0, 2, 1), kernel, padding=padding).permute(0, 2, 1)
    
    # Trim extra elements when window size is even
    if window % 2 == 0:
        smoothed_sequence = smoothed_sequence[:, :-1, :]

    return smoothed_sequence


def add_task_identity(inputs, task_index, num_tasks):
    """
    Extends the inputs tensor with the task identity.

    Args:
        inputs (torch.Tensor): The input tensor of shape (batch_size, time_steps, num_inputs).
        task_index (int): The index of the current task.
        num_tasks (int): The total number of tasks.

    Returns:
        torch.Tensor: The extended inputs tensor of shape (batch_size, time_steps, num_inputs + num_tasks).
    """
    
    # Convert task_index to one-hot encoding
    task_index = torch.tensor(task_index) if type(task_index) == int else task_index
    task_identity = F.one_hot(task_index, num_classes=num_tasks).float()

    # Expand task_identity to match the shape of inputs
    if len(inputs.shape) > 1: 
        expanded_task_identity = task_identity.unsqueeze(-2).expand(*inputs.shape[:-1], -1)
    else:
        expanded_task_identity = task_identity

    # Concatenate the expanded task_identity with inputs along the last dimension
    extended_inputs = torch.cat([inputs, expanded_task_identity], dim=-1)

    return extended_inputs