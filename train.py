import torch
import torch.nn.functional as F
import torch.optim as optim

import random
import math
import numpy as np
import json
import os
from tqdm import tqdm 

from utils.utils import get_hparams, get_model_path, load_checkpoint, get_metrics_path
from utils.model import MultitaskRNN
from utils.task import add_task_identity
import tasks

# Add your set_seed function here
def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def check_task_compatibility(tasks):
    """Check that all tasks have the same number of inputs and outputs."""
    num_inputs = tasks[0].num_inputs
    num_outputs = tasks[0].num_outputs

    for task in tasks[1:]:
        if task.num_inputs != num_inputs or task.num_outputs != num_outputs:
            return False

    return True

def compute_loss(rnn, tasks, task_idx, batch_size, period_duration, smoothing_window, grace_frac):
    with torch.no_grad():
        total_loss = 0
        num_sequences = batch_size

        task = tasks[task_idx]
        
        # Generate all input sequences for the task
        input_sequences, output_sequences = task.generate_batch(batch_size, period_duration, smoothing_window=smoothing_window)
        input_sequences = add_task_identity(input_sequences, task_idx, len(tasks))
            
        # Create mask for the validation set
        mask = task.generate_mask(period_duration, grace_frac).unsqueeze(0)
        mask = mask.repeat(num_sequences, 1, 1)   # Repeat the mask to match the number of sequences

        # Reset the hidden state
        hidden = rnn.init_hidden(num_sequences)

        # Forward pass on validation set
        predictions, hidden = rnn(input_sequences, hidden)  # assuming no need for the hidden state

        # Compute validation loss
        total_loss += F.mse_loss(predictions * mask, output_sequences * mask)

    return total_loss

def train_rnn_on_tasks(model_name, rnn, tasks, max_epochs, hparams):
    """Train the RNN on multiple tasks."""

    save_path = get_model_path(model_name)
    save_path_latest = get_model_path(model_name, latest=True)
    save_interval = hparams['save_interval']

    batch_size = hparams['batch_size']
    learning_rate = hparams['learning_rate']
    sigma_x = hparams['sigma_x']
    alpha = hparams['alpha']
    seed = hparams['seed']

    set_seed(seed)

    min_period = hparams['min_period']
    max_period = hparams['max_period']
    min_window = hparams['min_smoothing_window']
    max_window = hparams['max_smoothing_window']

    batch_size_test = hparams['batch_size_test']
    n_rep = hparams['test_n_rep']

    grace_frac = hparams['grace_frac']

    noise = sigma_x / math.sqrt(2 / alpha)

    # Check that all tasks have the same number of inputs and outputs
    if not check_task_compatibility(tasks):
        raise ValueError("All tasks must have the same number of inputs and outputs.")

    # Initialize the optimizer 
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Load the model from checkpoint if exists
    if os.path.isfile(save_path_latest):
        rnn, optimizer, start_epoch = load_checkpoint(model_name, rnn, optimizer, latest = True)
        print(f"Loaded checkpoint at epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Initialize or load performance metrics dictionary
    metrics_path = get_metrics_path(model_name)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            performance_metrics = json.load(f)
    else:
        performance_metrics = {"epochs": []}
        for i, task in enumerate(tasks):
          performance_metrics[f"task_{i}_loss"] = []



    # Initialize the progress bar
    progress_bar = tqdm(range(max_epochs))
    
    # Training loop
    for epoch in progress_bar:

        # Randomly select a task for the entire batch
        task_index = torch.randint(len(tasks), (1,)).item() 
        task = tasks[task_index]

        
        # Generate a batch of sequences for each selected task
        inputs = []
        outputs = []
        masks = []  # For masking the loss calculation

        # Get random period durations for each period
        period_durations = [random.randint(min_val, max_val + 1) for min_val, max_val in zip(min_period, max_period)]

        # Define the smoothing window
        smoothing_window = random.randint(min_window, max_window)

        # Select a single task for the whole batch
        task_index = torch.randint(len(tasks), (1,)).item()
        task = tasks[task_index]

        # Generate the full batch just for this task
        inputs, outputs = task.generate_batch(batch_size, period_durations, smoothing_window = smoothing_window, noise = noise)
        inputs = add_task_identity(inputs, task_index, len(tasks))

        # Create a mask for the grace period
        mask = task.generate_mask(period_durations, grace_frac).unsqueeze(0)
        masks = mask.repeat(batch_size, 1, 1)


        # Reset the hidden state
        hidden = rnn.init_hidden(batch_size)
        
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

        # Save the current state of the model as the 'latest' model
        model_data = {
            'model_state_dict': rnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + start_epoch,
            'tasks': tasks
        }
        torch.save(model_data, save_path_latest)
            
        if (epoch + start_epoch) % save_interval == 0:
            # Save the model every "save_interval" epochs
            interval_save_path = get_model_path(model_name, epoch=epoch+start_epoch)
            torch.save(model_data, interval_save_path)

            # Compute and print the loss
            for i in range(len(tasks)):
                task_losses = []
                for _ in range(n_rep):
                    batch_size = batch_size_test // n_rep
                    period_durations = [random.randint(min_val, max_val + 1) for min_val, max_val in zip(min_period, max_period)]
                    smoothing_window = random.randint(min_window, max_window)
                    task_loss = compute_loss(rnn, tasks, i, batch_size, period_durations, smoothing_window, grace_frac)
                    task_losses.append(task_loss.item())
                avg_task_loss = np.mean(task_losses)
                
                # Add current performance to metrics
                performance_metrics[f"task_{i}_loss"].append(avg_task_loss)
                
                # Update progress bar
                progress_bar.set_postfix({f'Task {i} Loss': avg_task_loss})
                
            # Add the epoch number to the metrics
            performance_metrics["epochs"].append(epoch + start_epoch)
            
            # Save performance metrics as a JSON file
            with open(metrics_path, "w") as f:
                json.dump(performance_metrics, f)            

    print("Training reached maximum epochs.")
        

tasks = [tasks.DelayGo(), tasks.DelayAnti()]

# Initialize RNN model
model_name = "delaygo_delayanti_var_durations"
hparams = get_hparams(model_name)

num_inputs = tasks[0].num_inputs + len(tasks)  # Include space for task identity inputs
num_outputs = tasks[0].num_outputs
num_hidden = hparams['num_hidden']

rnn = MultitaskRNN(num_inputs, num_hidden, num_outputs, hparams)

# Train the model
max_epochs = 30001

train_rnn_on_tasks(model_name, rnn, tasks, max_epochs, hparams)