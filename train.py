import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import random
import math
import numpy as np
import os
from tqdm import tqdm 

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

def compute_loss(rnn, tasks, period_duration, grace_frac):
    with torch.no_grad():
        total_loss = 0
        for task_index, task in enumerate(tasks):
            # Generate all input sequences for the task
            input_sequences, output_sequences = task.generate_all_sequences(period_duration=period_duration)
            input_sequences = add_task_identity(input_sequences, task_index, len(tasks))
            num_sequences = output_sequences.shape[0]

            # Create mask for the validation set
            mask = task.generate_mask(period_duration, grace_frac).unsqueeze(0)
            mask = mask.repeat(num_sequences, 1, 1)   # Repeat the mask to match the number of sequences

            # Reset the hidden state
            hidden = rnn.init_hidden(num_sequences)

            # Forward pass on validation set
            predictions, hidden = rnn(input_sequences, hidden)  # assuming no need for the hidden state

            # Compute validation loss
            total_loss += F.mse_loss(predictions * mask, output_sequences * mask)

        # Average the loss across tasks
        total_loss /= len(tasks)
    return total_loss

def train_rnn_on_tasks(model_name, rnn, tasks, max_epochs, hparams):
    """Train the RNN on multiple tasks."""

    save_path = get_model_path(model_name)
    save_path_latest = get_model_path(model_name, latest = True)
    save_interval = 500  # Save the model every 500 epochs
    
    batch_size = hparams['batch_size']
    learning_rate = hparams['learning_rate']
    sigma_x = hparams['sigma_x']
    alpha = hparams['alpha']

    min_period = 25
    max_period = 200

    validation_window = 1000
    val_threshold = 1e-3

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

    # Initialize best validation loss as infinity
    best_val_loss = float('inf')

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

        # Get a random period duration
        period_duration = random.randint(min_period, max_period + 1)

        for i in range(batch_size):
            
            # Define the smoothing window
            min_window = int(0.02 * period_duration) # 1
            max_window = int(0.4 * period_duration) # 20
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


        # Compute and print validation and average losses every 100 epochs
        if epoch % 100 == 0:
            
            val_loss = compute_loss(rnn, tasks, validation_window, grace_frac)
            avg_loss = compute_loss(rnn, tasks, int(0.5 * (min_period + max_period)), grace_frac)
            
            # Update progress bar
            progress_bar.set_postfix({'Validation Loss': val_loss.item(), 'Average Loss': avg_loss.item()})

            # If validation loss improved, save the model state
            if val_loss < best_val_loss:
                torch.save(model_data, save_path)

            # If validation loss below threshold, stop training
            if val_loss <= val_threshold:
                print(f"Validation loss is below the threshold of {val_threshold} at epoch {epoch + start_epoch}. Stopping training.")
                return
            
        if epoch % save_interval == 0:
            # Save the model every "save_interval" epochs
            interval_save_path = get_model_path(model_name, epoch=epoch+start_epoch)
            torch.save(model_data, interval_save_path)

    print("Training reached maximum epochs.")
        

# Initialize tasks
delay_go_task = DelayGo()
delay_anti_task = DelayAnti()

tasks = [delay_go_task, delay_anti_task]

# Initialize RNN model
model_name = "delaygo_delayanti_256"
hparams = get_hparams(model_name)

num_inputs = tasks[0].num_inputs + len(tasks)  # Include space for task identity inputs
num_outputs = tasks[0].num_outputs
num_hidden = hparams['num_hidden']

rnn = MultitaskRNN(num_inputs, num_hidden, num_outputs, hparams)

# Train the model
max_epochs = 100000

train_rnn_on_tasks(model_name, rnn, tasks, max_epochs, hparams)