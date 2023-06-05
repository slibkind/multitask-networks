import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base_task import Task
from utils import smooth_sequence


class DelayGo(Task):
    def __init__(self, period_duration=50, grace_period=0):
        
        # Generate the mask to allow a grace_period at the beginning of the go period
        mask = torch.ones(4 * period_duration, 2)
        mask[150:150+grace_period, :] = 0

        super().__init__(
            name="DelayGo",
            periods=["fixation", "stimulus_on", "delay", "go"],
            num_inputs=3,
            num_outputs=2,
            period_duration=period_duration,
            mask=mask
        )

    def generate_batch(self, batch_size, smoothing_window=1, noise = 0.0):
        sequence_length = self.num_periods * self.period_duration

        # Initialize a tensor to hold the input sequences
        inputs = torch.zeros(batch_size, sequence_length, 3)  # 3 input units: fixation, stimulus 1, stimulus 2

        # Initialize a tensor to hold the output sequences
        outputs = torch.zeros(batch_size, sequence_length, 2)  # 2 output units: stimulus 1, stimulus 2

        # Fill in the input and output sequences according to the task rules
        for i in range(batch_size):
            # Generate a random stimulus (1 or 2) for the "stimulus on" period
            stimulus_on = torch.randint(1, 3, (1,))

            for period in range(self.num_periods):
                if period == 0:  # fixation period
                    inputs[i, period*self.period_duration:(period+1)*self.period_duration, 0] = 1  # fixation input is on
                elif period == 1:  # stimulus on period
                    inputs[i, period*self.period_duration:(period+1)*self.period_duration, 0] = 1  # fixation input is on
                    inputs[i, period*self.period_duration:(period+1)*self.period_duration, stimulus_on] = 1  # the chosen stimulus is on
                elif period == 2:  # delay period
                    inputs[i, period*self.period_duration:(period+1)*self.period_duration, 0] = 1  # fixation input is on
                elif period == 3:  # go period
                    outputs[i, period*self.period_duration:(period+1)*self.period_duration, stimulus_on-1] = 1  # the chosen stimulus responds

        # Smooth the inputs
        inputs = smooth_sequence(inputs, smoothing_window)

        # Add noise to the inputs and outputs
        inputs += torch.randn_like(inputs) * noise

        return inputs, outputs

    def generate_all_sequences(self, smoothing_window=1, noise=0.0):
        sequence_length = self.num_periods * self.period_duration
        num_sequences = 2  # Number of sequences (stimulus_on = 1 or 2)

        # Initialize tensors for the input and output sequences
        inputs = torch.zeros(num_sequences, sequence_length, 3)  # 3 input units: fixation, stimulus 1, stimulus 2
        outputs = torch.zeros(num_sequences, sequence_length, 2)  # 2 output units: stimulus 1, stimulus 2

        for stimulus_on in range(1, num_sequences + 1):
            # Fill in the input and output sequences according to the task rules
            for period in range(self.num_periods):
                if period == 0:  # fixation period
                    inputs[stimulus_on - 1, period * self.period_duration:(period + 1) * self.period_duration, 0] = 1  # fixation input is on
                elif period == 1:  # stimulus on period
                    inputs[stimulus_on - 1, period * self.period_duration:(period + 1) * self.period_duration, 0] = 1  # fixation input is on
                    inputs[stimulus_on - 1, period * self.period_duration:(period + 1) * self.period_duration, stimulus_on] = 1  # the chosen stimulus is on
                elif period == 2:  # delay period
                    inputs[stimulus_on - 1, period * self.period_duration:(period + 1) * self.period_duration, 0] = 1  # fixation input is on
                elif period == 3:  # go period
                    outputs[stimulus_on - 1, period * self.period_duration:(period + 1) * self.period_duration, stimulus_on - 1] = 1  # the chosen stimulus responds
        
        # Smooth the inputs
        inputs = smooth_sequence(inputs, smoothing_window)
        
        # Add noise to the inputs and outputs
        inputs += torch.randn_like(inputs) * noise

        return inputs, outputs


         
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
            ax_row[1].plot(outputs[i, :, 0], label='Stimulus 1', color = "orange")
            ax_row[1].plot(outputs[i, :, 1], label='Stimulus 2', color = "green")
            ax_row[1].set_ylabel('Activation')
            ax_row[1].set_xlabel('Time')
            if i == 0:
                ax_row[1].legend()
                ax_row[1].set_title('Outputs')

        plt.subplots_adjust(hspace=1)    
        plt.tight_layout()
        plt.show()