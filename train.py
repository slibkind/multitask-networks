import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import random
import math
import numpy as np
import os

from utils.utils import get_hparams, get_model_path, load_checkpoint
from utils.model import MultitaskRNN
from utils.task import add_task_identity
from tasks import DelayGo, DelayAnti

# Add your set_seed function here
def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Set your seed here
seed = 2
set_seed(2) 

def check_task_compatibility(tasks):
    """Check that all tasks have the same number of inputs and outputs."""
    num_inputs = tasks[0].num_inputs
    num_outputs = tasks[0].num_outputs

    for task in tasks[1:]:
        if task.num_inputs != num_inputs or task.num_outputs != num_outputs:
            return False

    return True

def train_rnn_on_tasks(model_name, rnn, tasks, epochs, hparams):
    """Train the RNN on multiple tasks."""

    save_path = get_model_path(model_name)
    
    batch_size = hparams['batch_size']
    learning_rate = hparams['learning_rate']
    sigma_x = hparams['sigma_x']
    alpha = hparams['alpha']

    min_period = 25
    max_period = 200

    grace_frac = 0.1

    noise = sigma_x/math.sqrt(2/alpha)

    # Check that all tasks have the same number of inputs and outputs
    if not check_task_compatibility(tasks):
        raise ValueError("All tasks must have the same number of inputs and outputs.")

    # Initialize the optimizer and scheduler (if applicable)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Load the model from checkpoint if exists
    if os.path.isfile(save_path):
        rnn, optimizer, start_epoch = load_checkpoint(model_name, rnn, optimizer)
        print(f"Loaded checkpoint at epoch {start_epoch}")
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):

        # Reset the hidden state
        hidden = rnn.init_hidden(batch_size)

        # Randomly select a task for the entire batch
        task_index = torch.randint(len(tasks), (1,)).item() 
        task = tasks[task_index]

        
        # Generate a batch of sequences for each selected task
        inputs = []
        outputs = []
        masks = []  # For masking the loss calculation

        # Get a random period duration
        period_duration = random.randint(min_period, max_period + 1)

        for i in range(batch_size):
            
            # Define the smoothing window
            min_window = 1 #int(0.1 * period_duration)
            max_window = 20 #int(0.5 * period_duration)
            smoothing_window = random.randint(min_window, max_window)

            # Generate a batch of sequences for the task with the selected smoothing window
            task_inputs, task_outputs = task.generate_batch(1, period_duration = period_duration, smoothing_window=smoothing_window, noise = noise)
            
            # Extend the inputs with the task identity
            task_inputs = add_task_identity(task_inputs, task_index, len(tasks))
            
            inputs.append(task_inputs)
            outputs.append(task_outputs)

            # Create a mask for the grace period
            mask = task.generate_mask(period_duration, grace_frac).unsqueeze(0)
            masks.append(mask)

        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)
        masks = torch.cat(masks)

        # Forward pass
        predictions, hidden = rnn(inputs, hidden)

        # Compute the loss
        loss = F.mse_loss(predictions * masks, outputs * masks)

        # Add L1 and L2 regularization for weights
        l1_lambda = hparams['l1_lambda']  # Set your L1 regularization rate
        l2_lambda = hparams['l2_lambda']  # Set your L2 regularization rate

        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)

        for name, param in rnn.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1)
                l2_reg += torch.norm(param, 2)

        loss += l1_lambda * l1_reg + l2_lambda * l2_reg

        # Add L1 and L2 regularization for hidden states
        l1_h_lambda = hparams['l1_h_lambda']  # Set your L1 regularization rate for hidden states
        l2_h_lambda = hparams['l2_h_lambda']  # Set your L2 regularization rate for hidden states

        l1_h_reg = torch.norm(hidden, 1)
        l2_h_reg = torch.norm(hidden, 2)

        loss += l1_h_lambda * l1_h_reg + l2_h_lambda * l2_h_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=10)

        # Optimization
        optimizer.step()

        # Save the model at specified intervals
        if epoch % 100 == 0:
            model_data = {
                'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'tasks': tasks
            }
            torch.save(model_data, save_path)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

                
    print("Training complete.")
        

# Initialize tasks
delay_go_task = DelayGo()
delay_anti_task = DelayAnti()

tasks = [delay_go_task]

# Initialize RNN model
model_name = "delaygo_delayanti_64"
hparams = get_hparams(model_name)

num_inputs = tasks[0].num_inputs + len(tasks)  # Include space for task identity inputs
num_outputs = tasks[0].num_outputs
num_hidden = hparams['num_hidden']

rnn = MultitaskRNN(num_inputs, num_hidden, num_outputs, hparams)

# Train the model
epochs = 200

train_rnn_on_tasks(model_name, rnn, tasks, epochs, hparams)