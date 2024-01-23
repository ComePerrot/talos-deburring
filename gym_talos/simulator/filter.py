import numpy as np


class LowpassFilter:
    def __init__(self, cutoff_frequency, sampling_rate, num_channels):
        self.tau = 1 / (2 * np.pi * cutoff_frequency)
        self.alpha = self.tau / (self.tau + 1 / sampling_rate)

        self.num_channels = num_channels
        self.prev_output = np.zeros(num_channels)

    def filter(self, input_data):
        # Check input format
        if input_data.shape[0] != self.num_channels:
            msg = f"Input vector should have {self.num_channels} channels."
            raise ValueError(msg)

        output = self.alpha * input_data + (1 - self.alpha) * self.prev_output
        self.prev_output = output
        return output
