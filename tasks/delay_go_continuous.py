import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base_task import Task
from utils.task import smooth_sequence


import numpy as np

class DelayGoContinuous(Task):
    def __init__(self, N=8):
        super().__init__(
            name = "DelayGoContinuous",
            periods = ["fix", "stim", "delay", "go"],
            num_inputs = 3,
            num_outputs = 3
        )
        # N is the number of evenly spaced stimuli around the unit circle
        self.N = N
        # stimuli is an array of evenly spaced stimuli, from 0 to 2*pi (excluded)
        self.stimuli = np.linspace(0, 2*np.pi, N, endpoint=False)

    def get_input(self, period, stimulus):
        if stimulus < 0 or stimulus >= 2*np.pi:
            raise ValueError("Invalid stimulus value. Should be in [0, 2*pi).")

        if period not in ["fix", "stim", "delay", "go"]: 
            raise ValueError("Invalid period value. Should be one of 'fix', 'delay', 'stim', or 'go'.")

        input_tensor = torch.zeros(self.num_inputs)
        if period != "go":
            input_tensor[0] = 1  # Activate the fixation unit
        if period == "stim":
            input_tensor[1] = np.sin(stimulus)
            input_tensor[2] = np.cos(stimulus)

        return input_tensor

    def generate_batch(self, batch_size, period_duration = 50, smoothing_window=1, noise = 0.0):
        sequence_length = self.num_periods * period_duration

        inputs = torch.zeros(batch_size, sequence_length, self.num_inputs)
        outputs = torch.zeros(batch_size, sequence_length, self.num_outputs)

        for i in range(batch_size):
            stimulus = np.random.choice(self.stimuli)  # Select a random stimulus

            for period in range(self.num_periods):
                if period == 0:  # fixation period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                    outputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                elif period == 1:  # stimulus on period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                    inputs[i, period*period_duration:(period+1)*period_duration, 1] = np.sin(stimulus)
                    inputs[i, period*period_duration:(period+1)*period_duration, 2] = np.cos(stimulus)
                    outputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                elif period == 2:  # delay period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                    outputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                elif period == 3:  # go period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1
                    outputs[i, period*period_duration:(period+1)*period_duration, 1] = np.sin(stimulus)
                    outputs[i, period*period_duration:(period+1)*period_duration, 2] = np.cos(stimulus)

        inputs = smooth_sequence(inputs, smoothing_window)
        inputs += torch.randn_like(inputs) * noise

        return inputs, outputs

    def generate_all_sequences(self, period_duration = 50, smoothing_window=1, noise=0.0):
        sequence_length = self.num_periods * period_duration
        num_sequences = self.N  # Number of sequences

        inputs = torch.zeros(num_sequences, sequence_length, self.num_inputs)
        outputs = torch.zeros(num_sequences, sequence_length, self.num_outputs)

        for s, stimulus in enumerate(self.stimuli):
            for period in range(self.num_periods):
                if period == 0:  # fixation period
                    inputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                    outputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                elif period == 1:  # stimulus on period
                    inputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                    inputs[s, period*period_duration:(period+1)*period_duration, 1] = np.sin(stimulus)
                    inputs[s, period*period_duration:(period+1)*period_duration, 2] = np.cos(stimulus)
                    outputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                elif period == 2:  # delay period
                    inputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                    outputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                elif period == 3:  # go period
                    inputs[s, period*period_duration:(period+1)*period_duration, 0] = 1
                    outputs[s, period*period_duration:(period+1)*period_duration, 1] = np.sin(stimulus)
                    outputs[s, period*period_duration:(period+1)*period_duration, 2] = np.cos(stimulus)

        inputs = smooth_sequence(inputs, smoothing_window)
        inputs += torch.randn_like(inputs) * noise

        return inputs, outputs


    def generate_mask(self, period_duration = 50, grace_frac = 0):
        # Generate the mask to allow a grace period at the beginning of the go period
        # the grace period is a fraction of the period_duration

        mask = torch.ones(4 * period_duration, self.num_outputs)
        mask[3*period_duration:3*period_duration+int(period_duration*grace_frac), :] = 0
        return mask

         
    def plot_sequence(self, inputs, outputs):
        nrows = inputs.shape[0]
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows*2))

        for i, ax_row in enumerate(axes):
            # Plot inputs
            ax_row[0].plot(inputs[i, :, 0], label='Fixation')
            ax_row[0].plot(inputs[i, :, 1], label='Stimulus 1', color = "orange")
            ax_row[0].plot(inputs[i, :, 2], label='Stimulus 2', color = "green")
            ax_row[0].set_ylabel('Activation')
            ax_row[0].set_xlabel('Time')
            if i == 0: 
                ax_row[0].legend()
                ax_row[0].set_title('Inputs')

            # Plot outputs
            ax_row[1].plot(outputs[i, :, 0], label='Fixation')
            ax_row[1].plot(outputs[i, :, 1], label='Stimulus 1', color = "orange")
            ax_row[1].plot(outputs[i, :, 2], label='Stimulus 2', color = "green")
            ax_row[1].set_ylabel('Activation')
            ax_row[1].set_xlabel('Time')
            if i == 0:
                ax_row[1].legend()
                ax_row[1].set_title('Outputs')

        plt.subplots_adjust(hspace=1)    
        plt.tight_layout()
        plt.show()