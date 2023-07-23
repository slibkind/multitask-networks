import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base_task import Task
from utils.task import smooth_sequence


class DelayAnti(Task):
    def __init__(self, grace_frac=0.):

        super().__init__(
            name = "DelayAnti",
            periods = ["fix", "stim", "delay", "go"],
            num_inputs = 3,
            num_outputs = 3
        )


    def get_input(self, period, stimulus):
        """
        Returns the input corresponding to a given period and stimulus.

        Args:
            period (str): The period of the task. Should be one of ['fixation', 'stimulus', 'delay', 'go'].
            stimulus (int): The stimulus number. Should be either 1 or 2.

        Returns:
            torch.Tensor: The input tensor corresponding to the specified period and stimulus.

        Raises:
            ValueError: If the stimulus value is not 1 or 2.
            ValueError: If the period is not one of 'fix', 'delay', 'stim', 'go'

        """
        if stimulus not in [1, 2]:
            raise ValueError("Invalid stimulus value. Should be either 1 or 2.")
        
        if period not in ["fix", "stim", "delay", "go"]: 
            raise ValueError("Invalid period value. Should be one of 'fix', 'delay', 'stim', or 'go'.")

        input_tensor = torch.zeros(self.num_inputs)
        if period != "go":
            input_tensor[0] = 1  # Activate the fixation unit
        if period == "stim":
            input_tensor[stimulus] = 1

        return input_tensor

    def generate_mask(self, period_duration = 50, grace_frac = 0):
        # Generate the mask to allow a grace period at the beginning of the go period
        # the grace period is a fraction of the period_duration

        mask = torch.ones(4 * period_duration, self.num_outputs)
        mask[3*period_duration:3*period_duration+int(period_duration*grace_frac), :] = 0
        return mask
    
    def generate_batch(self, batch_size, period_duration = 50, smoothing_window=1, noise = 0.0):
        sequence_length = self.num_periods * period_duration

        # Initialize a tensor to hold the input sequences
        inputs = torch.zeros(batch_size, sequence_length, self.num_inputs)  # 3 input units: fixation, stimulus 1, stimulus 2

        # Initialize a tensor to hold the output sequences
        outputs = torch.zeros(batch_size, sequence_length, self.num_outputs)  # 3 output units: fixation, stimulus 1, stimulus 2

        # Fill in the input and output sequences according to the task rules
        for i in range(batch_size):
            # Generate a random stimulus (1 or 2) for the "stimulus on" period
            stimulus_on = torch.randint(1, 3, (1,))

            for period in range(self.num_periods):
                if period == 0:  # fixation period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1  # fixation input is on
                    outputs[i, period*period_duration:(period+1)*period_duration, 0] = 1  # the fixation output is on
                elif period == 1:  # stimulus on period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1  # fixation input is on
                    inputs[i, period*period_duration:(period+1)*period_duration, stimulus_on] = 1  # the opposite of the chosen stimulus is on
                    outputs[i, period*period_duration:(period+1)*period_duration, 0] = 1  # the fixation output is on
                elif period == 2:  # delay period
                    inputs[i, period*period_duration:(period+1)*period_duration, 0] = 1  # fixation input is on
                    outputs[i, period*period_duration:(period+1)*period_duration, 0] = 1  # the fixation output is on
                elif period == 3:  # go period
                    outputs[i, period*period_duration:(period+1)*period_duration, 3-stimulus_on] = 1  # the chosen stimulus responds

        # Smooth the inputs
        inputs = smooth_sequence(inputs, smoothing_window)

        # Add noise to the inputs and outputs
        inputs += torch.randn_like(inputs) * noise

        return inputs, outputs

    def generate_all_sequences(self, period_duration = 50, smoothing_window=1, noise=0.0):
        sequence_length = self.num_periods * period_duration
        num_sequences = 2  # Number of sequences (stimulus_on = 1 or 2)

        # Initialize tensors for the input and output sequences
        inputs = torch.zeros(num_sequences, sequence_length, self.num_inputs)  # 3 input units: fixation, stimulus 1, stimulus 2
        outputs = torch.zeros(num_sequences, sequence_length, self.num_outputs)  # 3 output units: fixation, stimulus 1, stimulus 2

        for stimulus_on in range(1, num_sequences + 1):
            # Fill in the input and output sequences according to the task rules
            for period in range(self.num_periods):
                if period == 0:  # fixation period
                    inputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 0] = 1  # fixation input is on
                    outputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 0] = 1  # the fixation output is on
                elif period == 1:  # stimulus on period
                    inputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 0] = 1  # fixation input is on
                    inputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, stimulus_on] = 1  # the chosen stimulus is on
                    outputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 0] = 1  # the fixation output is on
                elif period == 2:  # delay period
                    inputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 0] = 1  # fixation input is on
                    outputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 0] = 1  # the fixation output is on
                elif period == 3:  # go period
                    outputs[stimulus_on-1, period*period_duration:(period+1)*period_duration, 3-stimulus_on] = 1  # the chosen stimulus responds

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