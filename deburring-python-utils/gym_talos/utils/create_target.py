import itertools
from typing import Tuple

import numpy as np


class TargetGoal:
    """
    Create a target position for the environment

    :param _type_target: (str) type of target to create
    :param _range_target: (np.array) range of the target
        adapted to the target type
    :param _position_target: (np.array) position of the target
    :param _upperPositionLimit: (np.array) upper limit of the target
    :param _lowerPositionLimit: (np.array) lower limit of the target
    """

    def __init__(self, params_env: dict) -> None:
        self._type_target, self._range_target = self._sort_datas(params_env)
        self._upperPositionLimit = None
        self._lowerPositionLimit = None
        self._position_target = None
        self._init_scaling()

    def _init_scaling(self) -> None:
        """Initialize the scaling of the target"""
        if self._type_target == "fixed":
            self._upperPositionLimit = self._range_target + np.ones(3)
            self._lowerPositionLimit = self._range_target - np.ones(3)
        elif self._type_target == "box":
            self._lowerPositionLimit = self._range_target[:3] + self._range_target[3:6]
            self._upperPositionLimit = self._range_target[:3] + self._range_target[6:9]
        elif self._type_target == "sphere":
            self._upperPositionLimit = self._range_target[:3] + self._range_target[
                3
            ] * np.ones(3)
            self._lowerPositionLimit = self._range_target[:3] - self._range_target[
                3
            ] * np.ones(3)

    def create_sphere(self) -> np.ndarray:
        """
        Create a sphere target position for the environment

        :return: (np.array) position of the target
        """
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.arccos(np.random.uniform(-1, 1))
        u = np.random.uniform(0, self._range_target[3])
        return np.array(
            [
                self._range_target[0] + u * np.sin(theta) * np.cos(phi),
                self._range_target[1] + u * np.sin(theta) * np.sin(phi),
                self._range_target[2] + u * np.cos(theta),
            ],
        )

    def create_box(self) -> np.ndarray:
        """
        Create a box target position for the environment

        :return: (np.array) position of the target
        """
        size_low = self._range_target[3:6]
        size_high = self._range_target[6:9]

        return np.array(
            [
                self._range_target[0] + np.random.uniform(size_low[0], size_high[0]),
                self._range_target[1] + np.random.uniform(size_low[1], size_high[1]),
                self._range_target[2] + np.random.uniform(size_low[2], size_high[2]),
            ],
        )

    def create_target(self) -> None:
        """Create a target position for the environment"""

        if self._type_target == "fixed":
            self._position_target = self._range_target
        elif self._type_target == "box":
            self._position_target = self.create_box()
        elif self._type_target == "sphere":
            self._position_target = self.create_sphere()
        else:
            msg = "Unknown target type"
            raise ValueError(msg)

    def set_target(self, target) -> None:
        """Set the target position for the environment manually"""
        self._position_target = target

    def generate_target(self, n=1) -> np.ndarray:
        """Generate n target positions for the environment"""
        if self._type_target == "fixed":
            targets = np.tile(self._range_target, (n, 1))
        elif self._type_target == "box":
            targets = np.array([self.create_box() for _ in range(n)])
        elif self._type_target == "sphere":
            targets = np.array([self.create_sphere() for _ in range(n)])
        else:
            msg = "Unknown target type"
            raise ValueError(msg)
        return targets

    def _sort_datas(self, params_env: dict) -> Tuple[str, np.ndarray]:
        """
        Sort datas for the target

        :param params_env: (dict) parameters of the environment
        :return: (np.array) range adapted to the target type
        """
        t_type = params_env["targetType"].lower()
        if t_type == "fixed":
            return t_type, np.array(params_env["targetPosition"])
        if t_type == "box":
            return t_type, np.concatenate(
                (
                    np.array(params_env["targetPosition"]),
                    np.array(params_env["targetSizeLow"]),
                    np.array(params_env["targetSizeHigh"]),
                ),
            )
        if t_type == "sphere":
            return t_type, np.concatenate(
                (
                    np.array(params_env["targetPosition"]),
                    np.array([params_env["targetRadius"]]),
                ),
            )
        msg = "Unknown target type"
        raise ValueError(msg)

    def generate_target_list(self, sample_sizes):
        if self._type_target == "box":
            iterator_list = [
                np.linspace(
                    self._lowerPositionLimit[i],
                    self._upperPositionLimit[i],
                    sample_sizes[i],
                )
                for i in range(3)
            ]
            return list(itertools.product(*iterator_list))
        else:  # noqa: RET505
            msg = "Target type must be box for evaluation"
            raise ValueError(msg)

    @property
    def position_target(self):
        return self._position_target

    @property
    def type_target(self):
        return self._type_target

    @property
    def range_target(self):
        return self._range_target

    @property
    def upperPositionLimit(self):
        return self._upperPositionLimit

    @property
    def lowerPositionLimit(self):
        return self._lowerPositionLimit
