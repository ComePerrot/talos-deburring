from collections import deque

import numpy as np


class observation_wrapper:
    def __init__(
        self, normalize_obs, rmodel, target_handler, history_size, prediction_size,
    ):
        self.normalize_obs = normalize_obs
        self._init_normalizer(rmodel, target_handler)

        self.prediction_size = prediction_size

        self.queue_length = history_size + 1
        self.state_queue = deque(maxlen=self.queue_length)

        self.observation_size = (self.queue_length + self.prediction_size) * (
            rmodel.nq + rmodel.nv
        ) + 3

    def reset(self, x_measured, target_position, x_future_list):
        """Reset the observation wrapper

        Reset the state queue to have a coherant size of observation
        even for the first step

        Args:
            x_measured: measured state of the environment
            target_position: position of the target
        """
        if self.normalize_obs:
            x_measured = self.normalize_state(x_measured)
            self.target_position = self.normalize_target(target_position)
        else:
            self.target_position = target_position

        # Filling state queue with the first observation
        for _ in range(self.queue_length):
            self.state_queue.append(x_measured)

        observation = self._generate_observation_vector(x_future_list)
        return np.float32(observation)

    def get_observation(self, x_measured, x_future_list):
        """Generates environement observation array

        Generates the observations to be fed to the agent
        based on history of measures and target position

        Args:
            x_measured: measured state of the environment
            target_position: position of the target
        """
        if self.normalize_obs:
            x_measured = self.normalize_state(x_measured)

        self.state_queue.append(x_measured)

        observation = self._generate_observation_vector(x_future_list)
        return np.float32(observation)

    def _generate_observation_vector(self, x_future_list):
        if self.prediction_size > 0:
            observation = np.concatenate(
                (
                    np.array(self.state_queue).flatten(),
                    self.target_position,
                    self._generate_predicted_states(x_future_list),
                ),
            )
        else:
            observation = np.concatenate(
                (np.array(self.state_queue).flatten(), self.target_position),
            )

        return observation

    def _generate_predicted_states(self, x_future_list):
        horizon_length = len(x_future_list)
        predicted_states = [
            x_future_list[
                int((i + 1) * ((horizon_length - 1) / (self.prediction_size)))
            ]
            for i in range(self.prediction_size)
        ]
        if self.normalize_obs:
            predicted_states = self.normalize_state(predicted_states)

        return np.array(predicted_states).flatten()

    def _init_normalizer(self, rmodel, target_handler):
        """Initializes the state and target normalizers

        The state normalizer is initialized using robot model limits
        Target normalizer is initialized using the bound given by the target handler

        Args:
            rmodel: model of the robot
            target_handler: class that handles target generation
        """
        # State
        lower_state_lim = np.concatenate(
            (
                rmodel.lowerPositionLimit,
                -rmodel.velocityLimit,
            ),
        )
        lower_state_lim[:7] = -5
        lower_state_lim[rmodel.nq : rmodel.nq + 6] = -5

        upper_state_lim = np.concatenate(
            (
                rmodel.upperPositionLimit,
                rmodel.velocityLimit,
            ),
        )
        upper_state_lim[:7] = 5
        upper_state_lim[rmodel.nq : rmodel.nq + 6] = 5

        self.avg_state = (upper_state_lim + lower_state_lim) / 2
        self.diff_state = upper_state_lim - lower_state_lim

        # target
        lower_target_lim = target_handler.lowerPositionLimit
        upper_target_lim = target_handler.upperPositionLimit

        self.avg_target = (upper_target_lim + lower_target_lim) / 2
        self.diff_target = upper_target_lim - lower_target_lim

    def normalize_state(self, state):
        """Normalizes the given robot state taken from the simulator

        Args:
            state: robot state

        Returns:
            normalized state
        """

        return (state - self.avg_state) / self.diff_state

    def normalize_target(self, target):
        """Normalizes the given target position

        Args:
            target: 3d position of the target

        Returns:
            normalized target
        """
        return (target - self.avg_target) / self.diff_target
