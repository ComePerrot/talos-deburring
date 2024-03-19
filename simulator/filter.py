import numpy as np


class LowpassFilter:
    """A class to represent a first order low pass filter.

    Attributes:
        tau (float): The time constant of the filter.
        alpha (float): The smoothing factor of the filter.
        num_channels (int): The number of channels in the input data.
        prev_output (numpy.ndarray): The previous output of the filter.
    """

    def __init__(self, cutoff_frequency, sampling_rate, num_channels):
        """Initialize the LowpassFilter object.

        Args:
            cutoff_frequency: The cutoff frequency of the filter.
            sampling_rate: The sampling rate of the input data.
            num_channels: The number of channels in the input data.
        """
        self.tau = 1 / (2 * np.pi * cutoff_frequency)
        self.alpha = self.tau / (self.tau + 1 / sampling_rate)

        self.num_channels = num_channels
        self.prev_output = np.zeros(num_channels)

    def filter(self, input_data):
        """
        Apply the low pass filter to the input data.

        Args:
            input_data (numpy.ndarray): The input data to be filtered.
        """
        # Check input format
        if input_data.shape[0] != self.num_channels:
            msg = f"Invalid number of channels: {input_data.shape[0]} when {self.num_channels} is expected."
            raise ValueError(msg)

        output = self.alpha * input_data + (1 - self.alpha) * self.prev_output
        self.prev_output = output
        return output
