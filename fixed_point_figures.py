from utils.analysis import visualize_fixed_points
import matplotlib.pyplot as plt
import os

def plot_models(model_list, params_list, n_interp, q_thresh_list, save_path):
    for (model_name, epoch) in model_list:
        n_tasks = len(params_list)
        n_q_thresh = len(q_thresh_list)
        fig, axs = plt.subplots(n_q_thresh, n_tasks, figsize=(10*n_tasks, 8*n_q_thresh))
        fig.suptitle(f'Model: {model_name}', fontsize=16)

        for q_idx, q_thresh in enumerate(q_thresh_list):
            for t_idx, params in enumerate(params_list):
                task_idx = params['task_idx']
                period = params['period']
                stimulus = params['stimulus']
                labels = params['labels']
                title = params['title']

                ax = axs[q_idx, t_idx] if n_tasks > 1 else axs[q_idx]
                visualize_fixed_points(model_name, epoch, task_idx, period, stimulus, n_interp, 
                                       q_thresh=q_thresh, input_labels=labels, title=title, ax=ax)
                
                if t_idx == 0:  # if this is the first column
                    ax.set_ylabel(f'q_thresh: {q_thresh}')  # set the y-label

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  
        plt.savefig(os.path.join(save_path, f"{model_name}_{epoch}.pdf"))
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
        'labels': ["delay", "delay"],
        'title': f"Comparing the delay periods",
    },
    {
        'task_idx': [0, 1],
        'period': ["go", "go"],
        'stimulus': [1, 1],
        'labels': ["go", "go"],
        'title': f"Comparing the go periods",
    }
]

n_interp=20

save_path = "/Users/slibkind/Desktop/figures2"
model_list = [("delaygo_delayanti_var_durations", 55500)]
q_thresh_list = [1e-4, 1e-6, 1e-8]


plot_models(model_list, params_list, n_interp, q_thresh_list, save_path)  
