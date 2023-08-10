import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base_task import Task
from utils.task import smooth_sequence


class DelayAnti(Task):
    def __init__(self):

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

    def generate_mask(self, period_duration=50, grace_frac=0):
        """
        Generate a mask tensor for output sequences during training.

        Args:
            period_durations (int or tuple of int): Duration for each period. If an integer is provided, it is replicated for each period.
                                                    By default, set to 50 for all periods.
            grace_frac (float): Fraction of the 'go' period to be masked at the beginning.
                                This allows a grace period for the network response.

        Returns:
            torch.Tensor: A mask tensor with ones where the network output is expected and zeros during the grace period.
        """
        
        # Check if period_durations is an integer and replicate if needed
        if isinstance(period_duration, int):
            period_durations = (period_duration,) * self.num_periods

        total_duration = sum(period_duration)
        mask = torch.ones(total_duration, self.num_outputs)
        
        # Start of the 'go' period
        go_start = sum(period_duration[:-1])
        grace_duration = int(period_duration[-1] * grace_frac)

        mask[go_start:go_start+grace_duration, :] = 0
        return mask
    
    def generate_batch(self, batch_size, period_duration=50, smoothing_window=1, noise=0.0):
        """
        Generate a batch of input-output pairs for the DelayGo task.

        Args:
            batch_size (int): The number of sequences in the batch.
            period_duration (int or tuple of ints): Duration of each period in time steps. 
                - If an integer is provided, all periods will have the same duration.
                - If a tuple is provided, each period will have the duration specified by the corresponding tuple element.
            smoothing_window (int): The size of the smoothing window for the input sequences.
            noise (float): Standard deviation of Gaussian noise added to the inputs.

        Returns:
            tuple: A tuple containing the input and output sequences as torch tensors.
        """
        
        # Ensure period_duration is a tuple
        if isinstance(period_duration, int):
            period_duration = (period_duration,) * self.num_periods

        sequence_length = sum(period_duration)

        # Initialize a tensor to hold the input sequences
        inputs = torch.zeros(batch_size, sequence_length, self.num_inputs)  # 3 input units: fixation, stimulus 1, stimulus 2

        # Initialize a tensor to hold the output sequences
        outputs = torch.zeros(batch_size, sequence_length, self.num_outputs)  # 3 output units: fixation, stimulus 1, stimulus 2

        # Fill in the input and output sequences according to the task rules
        for i in range(batch_size):
            # Generate a random stimulus (1 or 2) for the "stimulus on" period
            stimulus_on = torch.randint(1, 3, (1,))

            start_time = 0
            for period, duration in enumerate(period_duration):
                end_time = start_time + duration
                if period == 0:  # fixation period
                    inputs[i, start_time:end_time, 0] = 1
                    outputs[i, start_time:end_time, 0] = 1
                elif period == 1:  # stimulus on period
                    inputs[i, start_time:end_time, 0] = 1
                    inputs[i, start_time:end_time, stimulus_on] = 1
                    outputs[i, start_time:end_time, 0] = 1
                elif period == 2:  # delay period
                    inputs[i, start_time:end_time, 0] = 1
                    outputs[i, start_time:end_time, 0] = 1
                elif period == 3:  # go period
                    outputs[i, start_time:end_time, 3-stimulus_on] = 1
                start_time = end_time

        # Smooth the inputs
        inputs = smooth_sequence(inputs, smoothing_window)

        # Add noise to the inputs
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