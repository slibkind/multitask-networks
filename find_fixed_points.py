import torch
import os
import time
import numpy as np
import random
import shutil

from utils.task import get_input
from utils.analysis import get_all_hiddens, minimize_speed, is_stable
from utils.utils import get_model, get_fixed_point_path, get_model_path, get_model_epoch, input_to_str

# Set seed for reproducibility
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def find_fixed_points(model_name, epoch, task_details, n_interp, learning_rate, grad_threshold, sample_proportion, save_interval=10):
    # Load model and get initial conditions for finding fixed points
    rnn, tasks = get_model(model_name, epoch)
    all_hiddens = get_all_hiddens(rnn, tasks)

    # Determine the number of samples from the proportion of hidden states
    hidden_state_count = all_hiddens.size(0)
    num_samples = int(hidden_state_count * sample_proportion)  # Calculate the number of samples

    # Randomly permute the indices and select the first num_samples
    indices = torch.randperm(hidden_state_count)[:num_samples]  
    sampled_hiddens = all_hiddens[indices]  # Select the sampled hidden points using the sampled indices

    # Generate task inputs for each pair of details
    fixed_point_path = get_fixed_point_path(model_name, epoch)
    
    # If the file already exists, load the existing data
    if os.path.exists(fixed_point_path):
        fixed_points_dict = torch.load(fixed_point_path)
    else:
        fixed_points_dict = {}  # Store all fixed points in a dictionary

    for task_idx, period, stimulus in task_details:
        
        # Generate the task inputs
        input1 = get_input(task_idx[0], period[0], stimulus[0], tasks)
        input2 = get_input(task_idx[1], period[1], stimulus[1], tasks)

        print(f"Task Details: {task_idx}, {period}, {stimulus}.")
        
        for i in range(n_interp + 1):
            print(f"\tInterpolation step {i+1} of {n_interp+1}")
            
            # Linearly interpolate between the inputs
            interpolated_input = (n_interp - i)/n_interp * input1 + (i/n_interp) * input2
        
            # Start timing
            start_time = time.time()
            
            # Find the fixed points for the interpolated_input
            fixed_points = minimize_speed(rnn, interpolated_input, sampled_hiddens, learning_rate,
                                          grad_threshold, verbose=False, method="second")
            
            # Compute the stability of each point
            stability = torch.tensor([is_stable(rnn, point, interpolated_input) for point in fixed_points], dtype=torch.bool)
  
            # Split fixed points into stable and unstable
            stable_points = fixed_points[stability]
            unstable_points = fixed_points[~stability]
              
            # End timing and print the execution time
            end_time = time.time()
            print(f"\t\tFinding fixed points took {end_time - start_time} seconds.")

            # Add the fixed points to the dictionary
            fixed_points_dict[input_to_str(interpolated_input)] = {"stable_points": stable_points, "unstable_points": unstable_points}
          
        # Save the dictionary of all fixed points
        torch.save(fixed_points_dict, fixed_point_path)
        print(f"Fixed points saved at {fixed_point_path}")




# Define task details (delaygo/delayanti)
task_details = [
    ([0, 0], ["stim", "delay"], [1, 1]),   # run through the delaygo task with stimulus 1
    ([0, 0], ["delay", "go"], [1, 1]),
    ([1, 1], ["stim", "delay"], [1, 1]),   # run through the delayanti task with stimulus 1
    ([1, 1], ["delay", "go"], [1, 1]),
    ([0, 0], ["stim", "delay"], [2, 2]),   # run through the delaygo task with stimulus 2
    ([1, 1], ["stim", "delay"], [2, 2]),   # run through the delayanti task with stimulus 2
    ([0, 1], ["delay", "delay"], [1, 1]),   # compare delay periods in delaygo vs. delayanti tasks
    ([0, 1], ["go", "go"], [1, 1])   # compare go periods in delaygo vs. delayanti tasks
]


# task_details = [
#     ([0, 0], ["stim", "delay"], [1, 1]),   # run through the delaygo task with stimulus 1
#     ([0, 0], ["delay", "go"], [1, 1]),
#     ([0, 0], ["stim", "delay"], [2, 2]),   # run through the delaygo task with stimulus 2
#     ([1, 1], ["stim_A", "delay"], [(1, 1), (1, 1)]), # run through the delay_match_to_sample task with stimulus 1, 1
#     ([1, 1], ["delay", "go"], [(1, 1), (1, 1)]),
#     ([1, 1], ["stim_A", "delay"], [(2, 2), (2, 2)]), # run through the delay_match_to_sample task with stimulus 1, 1
#     ([0, 1], ["delay", "delay"], [1, (1, 1)]),
#     ([0, 1], ["go", "go"], [1, (1, 1)])
# ]


# Model configuration
model_name = "delaygo_delayanti2"

# Interpolation steps
n_interp = 20

# Parameters for finding fixed points
learning_rate = 0.1
grad_threshold = 1e-3
sample_proportion = 0.1   # proportion of all hidden states to be sampled

# for epoch in range(5000, 20000, 5000):
epoch = 13500
find_fixed_points(model_name, epoch, task_details, n_interp, learning_rate, grad_threshold, sample_proportion)