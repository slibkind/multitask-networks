class Task:
    def __init__(self, name, periods, num_inputs, num_outputs, period_duration, mask):
        self.name = name
        self.periods = periods
        self.num_periods = len(periods)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.period_duration = period_duration
        self.mask = mask

    def generate_batch(self, batch_size, smooth_input=False):
        raise NotImplementedError

    def generate_all_sequences(self):
        raise NotImplementedError

    def plot_sequence(self, inputs, outputs):
        raise NotImplementedError