from utils.analysis import visualize_fixed_points
import matplotlib.pyplot as plt
import os

def plot_models(model_names, params_list, n_interp, save_path):
    for model_name in model_names:
        n_tasks = len(params_list)
        _, axs = plt.subplots(n_tasks, 1, figsize=(10, 8*n_tasks))
        for idx, params in enumerate(params_list):
            
            task_idx = params['task_idx']
            period = params['period']
            stimulus = params['stimulus']
            labels = params['labels']
            title = params['title']

            ax = axs[idx] if n_tasks > 1 else axs
            visualize_fixed_points(model_name, task_idx, period, stimulus, n_interp, 
                                   input_labels=labels, title=title, ax=ax)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{model_name}.pdf"))
        plt.close()



# use your function
params_list = [
    {
        'task_idx': [0,0,0,0],
        'period': ["fix", "stim", "delay", "go"],
        'stimulus': [1, 1, 1, 1],
        'labels': ["fix", "stim", "delay", "go"],
        'title': f"Dynamical landscape for DelayGo with stimulus 1",
    },
    {
        'task_idx': [0,0,0,0],
        'period': ["fix", "stim", "delay", "go"],
        'stimulus': [2, 2, 2, 2],
        'labels': ["fix", "stim", "delay", "go"],
        'title': f"Dynamical landscape for DelayGo with stimulus 2",
    },
    {
        'task_idx': [1, 1, 1, 1],
        'period': ["fix", "stim", "delay", "go"],
        'stimulus': [1, 1, 1, 1],
        'labels': ["fix", "stim", "delay", "go"],
        'title': f"Dynamical landscape for DelayAnti with stimulus 1",
    },
    {
        'task_idx': [1, 1, 1, 1],
        'period': ["fix", "stim", "delay", "go"],
        'stimulus': [2, 2, 2, 2],
        'labels': ["fix", "stim", "delay", "go"],
        'title': f"Dynamical landscape for DelayAnti with stimulus 2",
    },
    {
        'task_idx': [0, 1],
        'period': ["delay", "delay"],
        'stimulus': [1, 1],
        'labels': ["fix", "stim", "delay", "go"],
        'title': f"Comparing the delay periods",
    }
]

n_interp=20

save_path = "/Users/slibkind/Desktop/figures2"
model_names = ["delaygo_delayanti_255", "sophie delaygo_delayanti_256"]  # add your models here

      
plot_models(model_names, params_list, n_interp, save_path)  
