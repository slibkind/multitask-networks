import torch
import os

from utils.task import add_task_identity
from utils.analysis import get_all_hiddens, minimize_speed
from utils.utils import get_model, get_model_dir

model_name = "delaygo_delayanti_255"

task_idx = [0, 0]
period = ["stim", "delay"]
stimulus = [1, 2]

n_interp = 20

# parameters for finding fixed points
lr = 0.1
q_thresh = 1e-4
r = 0.1   # proportion of all hidden states to be sampled


def get_input(task_idx, period, stimulus, num_tasks):
    prelim_input = tasks[task_idx].get_input(period, stimulus)
    input = add_task_identity(prelim_input, task_idx, num_tasks)
    return input
 

# Get the analysis path
model_path = get_model_dir(model_name)
analysis_path = os.path.join(model_path, "analysis")

# Create the analysis directory if it doesn't already exist
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

# Get initial conditions for finding fixed points
rnn, tasks = get_model(model_name)
all_hiddens = get_all_hiddens(rnn, tasks)

k = all_hiddens.size(0)
num_samples = int(k * r)  # Calculate the number of samples as 10% of k

indices = torch.randperm(k)[:num_samples]  # Randomly permute the indices and select the first num_samples
sampled_hiddens = all_hiddens[indices]  # Select the sampled hidden points using the sampled indices


input1 = get_input(task_idx[0], period[0], stimulus[0], len(tasks))
input2 = get_input(task_idx[1], period[1], stimulus[1], len(tasks))

for i in range(n_interp + 1):
    
    # Linearly interpolate between the inputs
    input = (n_interp - i)/n_interp * input1 + (i/n_interp) * input2

    # Find the fixed points for the input
    fixed_points = minimize_speed(rnn, input, sampled_hiddens, lr, q_thresh)

    # Save the fixed points
    input_str = "_".join([str(int(t)) for t in 1000*input])
    torch.save(fixed_points, os.path.join(analysis_path, f'fixed_points_{input_str}.pt'))