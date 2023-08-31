from collections import deque

import numpy as np


class observation_wrapper:
    def __init__(self, normalize_obs, rmodel, target_handler, history_size):
        self.normalize_obs = normalize_obs
        self._init_normalizer(rmodel, target_handler)

        self.queue_length = history_size + 1
        self.state_queue = deque(maxlen=self.queue_length)

        self.size = self.queue_length * (rmodel.nq + rmodel.nv) + 3

    def reset(self, x_measured, target_position):
        if self.normalize_obs:
            x_measured = self.normalize_state(x_measured)
            target_position = self.normalize_target(target_position)
        for _ in range(self.queue_length):
            self.state_queue.append(x_measured)
        observation = np.concatenate(
            (np.array(self.state_queue).flatten(), target_position),
        )
        return np.float32(observation)

    def generate_observation(self, x_measured, target_position):
        if self.normalize_obs:
            x_measured = self.normalize_state(x_measured)
            target_position = self.normalize_target(target_position)
        self.state_queue.append(x_measured)
        observation = np.concatenate(
            (np.array(self.state_queue).flatten(), target_position),
        )
        return np.float32(observation)

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
        lower_state_lim[
            self.pinWrapper.get_rmodel().nq : self.pinWrapper.get_rmodel().nq + 6
        ] = -5

        upper_state_lim = np.concatenate(
            (
                rmodel.upperPositionLimit,
                rmodel.velocityLimit,
            ),
        )
        upper_state_lim[:7] = 5
        upper_state_lim[
            self.pinWrapper.get_rmodel().nq : self.pinWrapper.get_rmodel().nq + 6
        ] = 5

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
