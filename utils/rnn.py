import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import os

from utils.sequences import add_task_identity

model_path = 'models/'

import os

def get_model_path(task_names, num_hidden, hparams):
    alpha = hparams['alpha']
    activation = hparams['activation']
    model_name = '-'.join(task_names)
    model_name += f'_hidden{num_hidden}'
    model_name += f'_alpha{int(alpha * 100)}'
    model_name += f'_{activation}'
    model_name += '.pt'

    save_path = os.path.join(model_path, model_name)  # Assuming model_path is defined
    return save_path


def get_model_checkpoint(task_names, num_hidden, hparams):
    model_path = get_model_path(task_names, num_hidden, hparams)
    return torch.load(model_path)


def get_model(task_names, num_hidden, hparams):
    model_checkpoint = get_model_checkpoint(task_names, num_hidden, hparams)
    
    model_state_dict = model_checkpoint['model_state_dict']
    tasks = model_checkpoint['tasks']

    # Create the MultitaskRNN instance
    rnn = MultitaskRNN(input_size=tasks[0].num_inputs + len(tasks),
                       hidden_size=num_hidden,
                       output_size=tasks[0].num_outputs,
                       hparams=hparams)
    rnn.load_state_dict(model_state_dict)
    return rnn, tasks



def __init__(self, input_size, hidden_size, output_size, hparams):
    super(MultitaskRNN, self).__init__()
    
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.hparams = hparams
    self.alpha = hparams['alpha']
    self.sigma = torch.sqrt(torch.tensor(2.0 / self.alpha)) * hparams['sigma_rec']
    self.activation = self.get_activation(hparams['activation'])

    # Initialize weights and biases
    self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) / torch.sqrt(torch.tensor(float(input_size))) * hparams['w_in_start'])
    self.W_rec = self.initialize_W_rec(hidden_size)
    self.b = nn.Parameter(torch.randn(hidden_size))

    self.W_out = nn.Parameter(init.xavier_uniform_(torch.empty(output_size, hidden_size)))
    self.b_out = nn.Parameter(torch.randn(output_size))


class MultitaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hparams):
        super(MultitaskRNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self.alpha = hparams['alpha']
        self.sigma = torch.sqrt(torch.tensor(2.0 / self.alpha)) * hparams['sigma_rec']
        self.activation = self.get_activation(hparams['activation'])

        # Initialize weights and biases
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) / torch.sqrt(torch.tensor(float(input_size))) * hparams['w_in_coeff'])
        self.W_rec = self.initialize_W_rec(hidden_size)
        self.b = nn.Parameter(torch.zeros(hidden_size))

        self.W_out = nn.Parameter(init.xavier_uniform_(torch.empty(output_size, hidden_size)))
        self.b_out = nn.Parameter(torch.randn(output_size))
        
    def initialize_W_rec(self, hidden_size):
        if self.hparams['w_rec_init'] == 'diag':
            w_rec = torch.eye(hidden_size) * self.hparams['w_rec_coeff']
        elif self.hparams['w_rec_init'] == 'randgauss':
            w_rec = self.hparams['w_rec_coeff'] * torch.randn(hidden_size, hidden_size) / torch.sqrt(torch.tensor(float(hidden_size)))
        else:
            raise ValueError('Unknown w_rec_init')
        
        return nn.Parameter(w_rec)

    def get_activation(self, activation):
        if activation == 'softplus':
            return torch.nn.Softplus()
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'relu':
            return torch.relu
        elif activation == 'power':
            return lambda x: torch.pow(torch.relu(x), 2)
        elif activation == 'retanh':
            return lambda x: torch.tanh(torch.relu(x))
        else:
            raise ValueError('Unknown activation')


    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        outputs = torch.zeros(batch_size, inputs.size(1), self.output_size)  # Adjust output_size based on your model

        # Iterate over the time steps of the input sequence
        for i, input_t in enumerate(inputs.unbind(dim=1)):
            # Add Gaussian noise to each hidden state update
            private_noise = torch.randn_like(hidden) * self.sigma

            # Implement the RNN equation
            hidden = (1 - self.alpha) * hidden + self.alpha * self.activation(torch.einsum('ij,bj->bi', self.W_in, input_t) + torch.einsum('ij,bj->bi', self.W_rec, hidden) + self.b + private_noise) 
            outputs[:, i] = torch.einsum('ij,bj->bi', self.W_out, hidden) + self.b_out

        return outputs, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state
        return torch.zeros(batch_size, self.hidden_size)


def run_model(rnn, tasks, task_index):
    """
    Runs the RNN model on the input sequences for a specific task and returns the inputs, outputs, output trajectory,
    and hidden trajectory.

    Args:
        rnn (nn.Module): The RNN model.
        tasks (list): List of tasks.
        task_index (int): Index of the task to run the model on.

    Returns:
        tuple: A tuple containing the following elements:
            - inputs (torch.Tensor): The input sequences for the specified task.
            - outputs (torch.Tensor): The corresponding output sequences for the specified task.
            - output_trajectory (torch.Tensor): The output trajectory of the model during the run.
            - hidden_trajectory (torch.Tensor): The hidden state trajectory of the model during the run.
    """

    task = tasks[task_index]

    # Generate all input sequences for the task
    input_sequences, output_sequences = task.generate_all_sequences()
    input_sequences = add_task_identity(input_sequences, task_index, len(tasks))

    # Run the network with the input sequences in batches
    batch_size = input_sequences.shape[0]
    hidden = rnn.init_hidden(batch_size=batch_size)

    # Initialize the output and hidden trajectories with zeros
    output_trajectory = torch.zeros(batch_size, input_sequences.size(1), rnn.output_size)  # Adjust output_size based on your model
    hidden_trajectory = torch.zeros(batch_size, input_sequences.size(1), rnn.hidden_size)  # Adjust hidden_size based on your model

    # Iterate over the time steps of the input sequence
    for i, input_t in enumerate(input_sequences.unbind(dim=1)):
        # Implement the RNN equation
        hidden = (1 - rnn.alpha) * hidden + rnn.alpha * rnn.activation(torch.einsum('ij,bj->bi', rnn.W_in, input_t) + torch.einsum('ij,bj->bi', rnn.W_rec, hidden) + rnn.b)
        output = torch.einsum('ij,bj->bi', rnn.W_out, hidden) + rnn.b_out

        # Update the trajectories with the current output and hidden states
        output_trajectory[:, i] = output
        hidden_trajectory[:, i] = hidden

    return input_sequences, output_sequences, output_trajectory, hidden_trajectory

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

