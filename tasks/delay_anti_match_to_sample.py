import torch
from .base_task import Task
from utils.task import smooth_sequence
import matplotlib.pyplot as plt


class DelayAntiMatchToSample(Task):
    def __init__(self):
        super().__init__(
            name="DelayAntiMatchToSample",
            periods=["fix", "stim_A", "delay_1", "stim_B", "delay_2", "go"],
            num_inputs=3,
            num_outputs=3
        )

    def get_input(self, period, stimulus_A, stimulus_B):
        """
        Returns the input corresponding to a given period and stimuli.

        Args:
            period (str): The period of the task. Should be one of ['fix', 'stim_A', 'delay_1', 'stim_B', 'delay_2', 'go'].
            stimulus_A (int): The stimulus number for the 'stim_A' period. Should be either 1 or 2.
            stimulus_B (int): The stimulus number for the 'stim_B' period. Should be either 1 or 2.

        Returns:
            torch.Tensor: The input tensor corresponding to the specified period and stimuli.

        Raises:
            ValueError: If the stimulus values are not 1 or 2.
            ValueError: If the period is not valid.
        """
        if stimulus_A not in [1, 2] or stimulus_B not in [1, 2]:
            raise ValueError("Invalid stimulus value(s). Should be either 1 or 2.")

        valid_periods = ["fix", "stim_A", "delay_1", "stim_B", "delay_2", "go"]
        if period not in valid_periods:
            raise ValueError(f"Invalid period value. Should be one of {valid_periods}.")

        input_tensor = torch.zeros(self.num_inputs)
        if period != "go":
            input_tensor[0] = 1  # Activate the fixation unit
        if period == "stim_A":
            input_tensor[stimulus_A] = 1
        elif period == "stim_B":
            input_tensor[stimulus_B] = 1

        return input_tensor


    def generate_mask(self, period_duration=50, grace_frac=0):
        # Generate the mask to allow a grace period at the beginning of the go period
        # the grace period is a fraction of the period_duration

        mask = torch.ones(6 * period_duration, self.num_outputs)
        mask[5 * period_duration:5 * period_duration + int(period_duration * grace_frac), :] = 0
        return mask

    def generate_batch(self, batch_size, period_duration=50, smoothing_window=1, noise=0.0):
        sequence_length = self.num_periods * period_duration

        # Initialize a tensor to hold the input sequences
        inputs = torch.zeros(batch_size, sequence_length, self.num_inputs)

        # Initialize a tensor to hold the output sequences
        outputs = torch.zeros(batch_size, sequence_length, self.num_outputs)

        # Fill in the input and output sequences according to the task rules
        for i in range(batch_size):
            # Generate random stimuli (1 or 2) for "stimulus A on" and "stimulus B on" periods
            stim_A = torch.randint(1, 3, (1,))
            stim_B = torch.randint(1, 3, (1,))

            for period in range(self.num_periods):
                if period in [0, 1, 2, 3, 4]:  # fixation, stim_A, delay_1, stim_B, delay_2 periods
                    inputs[i, period * period_duration:(period + 1) * period_duration, 0] = 1  # fixation input is on
                    outputs[i, period * period_duration:(period + 1) * period_duration, 0] = 1  # the fixation output is on

                if period == 1:  # stim_A period
                    inputs[i, period * period_duration:(period + 1) * period_duration, stim_A] = 1  # the chosen stimulus A is on

                elif period == 3:  # stim_B period
                    inputs[i, period * period_duration:(period + 1) * period_duration, stim_B] = 1  # the chosen stimulus B is on

                elif period == 5:  # go period
                    if stim_A != stim_B:  # if the stimuli do not match
                        outputs[i, period * period_duration:(period + 1) * period_duration, stim_B] = 1  # the chosen stimulus B responds

        # Smooth the inputs
        inputs = smooth_sequence(inputs, smoothing_window)

        # Add noise to the inputs and outputs
        inputs += torch.randn_like(inputs) * noise

        return inputs, outputs

    def generate_all_sequences(self, period_duration=50, smoothing_window=1, noise=0.0):
        sequence_length = self.num_periods * period_duration
        num_sequences = 4  # Number of sequences (stim_A, stim_B = 1 or 2)

        # Initialize tensors for the input and output sequences
        inputs = torch.zeros(num_sequences, sequence_length, self.num_inputs)
        outputs = torch.zeros(num_sequences, sequence_length, self.num_outputs)

        # Define all possible combinations of stim_A and stim_B
        stimulus_combinations = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for idx, (stim_A, stim_B) in enumerate(stimulus_combinations):
            for period in range(self.num_periods):
                if period in [0, 1, 2, 3, 4]:  # fixation, stim_A, delay_1, stim_B, delay_2 periods
                    inputs[idx, period * period_duration:(period + 1) * period_duration, 0] = 1  # fixation input is on
                    outputs[idx, period * period_duration:(period + 1) * period_duration, 0] = 1  # the fixation output is on

                if period == 1:  # stim_A period
                    inputs[idx, period * period_duration:(period + 1) * period_duration, stim_A] = 1  # the chosen stimulus A is on

                elif period == 3:  # stim_B period
                    inputs[idx, period * period_duration:(period + 1) * period_duration, stim_B] = 1  # the chosen stimulus B is on

                elif period == 5:  # go period
                    if stim_A != stim_B:  # if the stimuli do not match
                        outputs[idx, period * period_duration:(period + 1) * period_duration, stim_B] = 1  # the chosen stimulus B responds

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
