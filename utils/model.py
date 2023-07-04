import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import matplotlib.pyplot as plt

from utils.task import add_task_identity
from utils.utils import get_model




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
    
    def forward_trajectory(self, inputs, hidden):
        """
        Run the RNN forward for the input sequence and return the output and hidden states trajectories.

        Args:
            inputs (torch.Tensor): The inputs to the RNN. This should be a 3D tensor of shape (batch_size, timesteps, n_inputs).
            hidden (torch.Tensor): The initial hidden states for the RNN. This should be a 2D tensor of shape (batch_size, n_hidden).

        Returns: 
            torch.Tensor, torch.Tensor: The output trajectory and hidden state trajectory. Both are 3D tensors of shape (batch_size, timesteps, feature_size).
        """
        batch_size = inputs.size(0)
        timesteps = inputs.size(1)
        outputs = torch.zeros(batch_size, timesteps, self.output_size)  # Adjust output_size based on your model
        hidden_states = torch.zeros(batch_size, timesteps, hidden.size(1)) # Tensor to store hidden states at each time-step

        # Iterate over the time steps of the input sequence
        for i, input_t in enumerate(inputs.unbind(dim=1)):
            # Implement the RNN equation
            hidden = (1 - self.alpha) * hidden + self.alpha * self.activation(torch.einsum('ij,bj->bi', self.W_in, input_t) + torch.einsum('ij,bj->bi', self.W_rec, hidden) + self.b) 
            outputs[:, i] = torch.einsum('ij,bj->bi', self.W_out, hidden) + self.b_out
            hidden_states[:, i] = hidden

        return outputs, hidden_states


def run_model(rnn, tasks, task_index, period_duration=50):
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
    input_sequences, output_sequences = task.generate_all_sequences(period_duration=period_duration)
    input_sequences = add_task_identity(input_sequences, task_index, len(tasks))

    # Initialize the hidden state
    hidden = rnn.init_hidden(batch_size=input_sequences.size(0))

    # Run the forward_trajectory method to get the output and hidden trajectories
    output_trajectory, hidden_trajectory = rnn.forward_trajectory(input_sequences, hidden)

    return input_sequences, output_sequences, output_trajectory, hidden_trajectory



def plot_behavior(model_name, period_duration=50):
    """
    Plot the behavior of a trained recurrent neural network (RNN) on different tasks.

    Parameters:
        model_name (str): The name of the trained model.
        period_duration (int, optional): The duration of each period in the input sequences. Default is 50.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified model file does not exist.

    """

    # Retrieve the model and task information
    rnn, tasks = get_model(model_name)
    
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
