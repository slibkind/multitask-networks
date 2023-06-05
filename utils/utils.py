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