import torch
import os
import time
import numpy as np
import random

from utils.task import get_input
from utils.analysis import get_all_hiddens, minimize_speed
from utils.utils import get_model, get_analysis_path, get_fixed_point_path

# Set seed for reproducibility
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Model configuration
model_name = "delaygo_delayanti_255"

# Define task details
task_details = [
    ([0, 0], ["stim", "delay"], [1, 1]),    # run through the delaygo task with stimulus 1
    ([0, 0], ["delay", "go"], [1, 1]),
    ([0, 0], ["delay", "delay"], [1, 2]), # compare stim 1 and 2 in delaygo task
    ([0, 0], ["stim", "delay"], [2, 2]),
    ([0, 1], ["delay", "delay"], [1, 1]),  # compare delaygo vs. delayanti tasks
    ([0, 1], ["delay", "delay"], [2, 2]),    
    ([1, 1], ["stim", "delay"], [1, 1]),
    ([1, 1], ["stim", "delay"], [2, 2])
]


# Interpolation steps
n_interp = 20

# Parameters for finding fixed points
learning_rate = 0.1
speed_threshold = 1e-6
sample_proportion = 0.1   # proportion of all hidden states to be sampled

# Get the analysis path
analysis_path = get_analysis_path(model_name)

# Create the analysis directory if it doesn't already exist
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

# Load model and get initial conditions for finding fixed points
rnn, tasks = get_model(model_name)
all_hiddens = get_all_hiddens(rnn, tasks)

# Determine the number of samples from the proportion of hidden states
hidden_state_count = all_hiddens.size(0)
num_samples = int(hidden_state_count * sample_proportion)  # Calculate the number of samples

# Randomly permute the indices and select the first num_samples
indices = torch.randperm(hidden_state_count)[:num_samples]  
sampled_hiddens = all_hiddens[indices]  # Select the sampled hidden points using the sampled indices

# Generate task inputs for each pair of details
for task_idx, period, stimulus in task_details:
    
    # Generate the task inputs
    input1 = get_input(task_idx[0], period[0], stimulus[0], tasks)
    input2 = get_input(task_idx[1], period[1], stimulus[1], tasks)
    
    for i in range(n_interp + 1):
        
        print(f"Task Details: {task_idx}, {period}, {stimulus}. Interpolation step {i+1} of {n_interp+1}")
        
        # Linearly interpolate between the inputs
        interpolated_input = (n_interp - i)/n_interp * input1 + (i/n_interp) * input2
    
        # Start timing
        start_time = time.time()
        
        # Find the fixed points for the interpolated_input
        fixed_points = minimize_speed(rnn, interpolated_input, sampled_hiddens, learning_rate,
                                      speed_threshold, verbose=True, method="second")
    
        # End timing and print the execution time
        end_time = time.time()
        print(f"Finding fixed points took {end_time - start_time} seconds.")
    
        # Save the fixed points
        fixed_point_path = get_fixed_point_path(model_name, interpolated_input) 
        torch.save(fixed_points, fixed_point_path)
    
        print(f"Fixed points saved at {fixed_point_path}\n")
