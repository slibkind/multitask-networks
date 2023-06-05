import torch
from torch import nn
import torch.nn.functional as F

class MultitaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, alpha, activation_function):
        super(MultitaskRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.activation = activation_function

        # Initialize weights and biases
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))

        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size))
        self.b_out = nn.Parameter(torch.randn(output_size))

    def forward(self, inputs, hidden):
        outputs = []
        # Iterate over the time steps of the input sequence
        for input_t in inputs.unbind(dim=1):
            # Implement the RNN equation
            hidden = (1 - self.alpha) * hidden + self.alpha * self.activation(torch.einsum('ij,bj->bi', self.W_in, input_t) + torch.einsum('ij,bj->bi', self.W_rec, hidden) + self.b)
            outputs.append(torch.einsum('ij,bj->bi', self.W_out, hidden) + self.b_out)
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state
        return torch.zeros(batch_size, self.hidden_size)
    
def get_model_name(tasks, num_hidden, alpha, activation):
    task_names = [task.__class__.__name__ for task in tasks]
    model_name = '-'.join(task_names)
    model_name += f'_hidden{num_hidden}'
    model_name += f'_alpha{int(alpha*100)}'
    model_name += f'_{activation.__name__}'
    model_name += '.pt'
    return model_name

