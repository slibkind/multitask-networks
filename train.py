import torch
import torch.nn.functional as F

import os
import random

from utils.rnn import MultitaskRNN, get_model_name
from tasks import DelayGo

model_path = "models/"

def check_task_compatibility(tasks):
    """Check that all tasks have the same number of inputs and outputs."""
    num_inputs = tasks[0].num_inputs
    num_outputs = tasks[0].num_outputs

    for task in tasks[1:]:
        if task.num_inputs != num_inputs or task.num_outputs != num_outputs:
            return False

    return True

def train_rnn_on_tasks(rnn, tasks, epochs, batch_size, learning_rate, patience):
    """Train the RNN on multiple tasks."""

    noise = 0.05

    # Generate a descriptive model name and create the save path
    model_name = get_model_name(tasks, num_hidden, alpha, activation)
    save_path = os.path.join(model_path, model_name)

    # Check that all tasks have the same number of inputs and outputs
    if not check_task_compatibility(tasks):
        raise ValueError("All tasks must have the same number of inputs and outputs.")

    # Initialize the optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    best_loss = float('inf')  # Initialize the best loss to a large value
    no_improvement_count = 0  # Count the number of epochs without improvement

    # Training loop
    for epoch in range(epochs):
        # Reset the hidden state
        hidden = rnn.init_hidden(batch_size)

        # Randomly select a task for each sequence in the batch
        task_indices = torch.randint(len(tasks), (batch_size,))
        
        # Generate a batch of sequences for each selected task
        inputs = []
        outputs = []
        masks = []  # For masking the loss calculation
        for i in range(batch_size):
            task = tasks[task_indices[i]]

            # Define the smoothing window
            min_window = int(0.1 * task.period_duration)
            max_window = int(0.5 * task.period_duration)
            smoothing_window = random.randint(min_window, max_window)

            # Generate a batch of sequences for the task with the selected smoothing window
            task_inputs, task_outputs = task.generate_batch(1, smoothing_window=smoothing_window, noise = noise)

            
            # Extend the inputs with the task identity
            task_identity = F.one_hot(task_indices[i], num_classes=len(tasks)).float()
            task_inputs = torch.cat([task_inputs, task_identity.expand(task_inputs.shape[:2] + (-1,))], dim=-1)
            inputs.append(task_inputs)
            outputs.append(task_outputs)

            # Create a mask for the grace period
            mask = task.mask.unsqueeze(0)
            masks.append(mask)

        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)
        masks = torch.cat(masks)

        # Forward pass
        predictions, hidden = rnn(inputs, hidden)

        # Compute the loss
        loss = F.mse_loss(predictions * masks, outputs * masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the model at specified intervals
        if epoch % 100 == 0:
            model_data = {
                'model_state_dict': rnn.state_dict(),
                'tasks': tasks
            }
            torch.save(model_data, save_path)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

            # Apply early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                
    print("Training complete.")
        

# Initialize tasks
delay_go_task = DelayGo(period_duration=50)
# delay_anti_task = DelayAntiTask(period_duration=50)
tasks = [delay_go_task]

# Initialize RNN model
num_inputs = tasks[0].num_inputs + len(tasks)  # Include space for task identity inputs
num_outputs = tasks[0].num_outputs
num_hidden = 3
alpha = 0.1
activation = torch.tanh

rnn = MultitaskRNN(num_inputs, num_hidden, num_outputs, alpha, activation)

# Train the model
epochs = 10000
batch_size = 32
learning_rate = 0.01
patience = 10

train_rnn_on_tasks(rnn, tasks, epochs, batch_size, learning_rate, patience)