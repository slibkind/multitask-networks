import matplotlib.pyplot as plt
from utils.rnn import MultitaskRNN, get_model, run_model
from utils.sequences import add_task_identity
import torch
import torch.nn.functional as F

def plot_behavior(task_names, num_hidden, alpha, activation):
    
    # Retrieve the model and task information
    rnn, tasks = get_model(task_names, num_hidden, alpha, activation)
    
    # Iterate over each task
    for task_index in range(len(tasks)):
        task_name = tasks[task_index].__class__.__name__
        inputs, outputs, output_pred, _ = run_model(rnn, tasks, task_index)

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
