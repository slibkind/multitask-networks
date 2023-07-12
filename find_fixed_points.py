#%%
import torch
import os
import time

from utils.task import get_input
from utils.analysis import get_all_hiddens, minimize_speed
from utils.utils import get_model, get_analysis_path, get_fixed_point_path

# Model configuration
model_name = "delaygo_delayanti_255"

# Define task details
task_idx = [0, 0]
period = ["stim", "delay"]
stimulus = [1, 2]

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

# Generate task inputs
input1 = get_input(task_idx[0], period[0], stimulus[0], tasks)
input2 = get_input(task_idx[1], period[1], stimulus[1], tasks)

for i in range(n_interp + 1):
    
    print(f"Interpolation step {i+1} of {n_interp+1}")
    
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

# %%
