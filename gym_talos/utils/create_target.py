import numpy as np


class TargetGoal:
    """
    Create a target position for the environment

    :param type: (str) type of target to create
    :param range: (np.array) range of the target
        adapted to the target type
    :param position: (np.array) position of the target
    """

    def __init__(self, params_env):
        self._type_target, self._range_target = self._sort_datas(params_env)
        self._upperPositionLimit = None
        self._lowerPositionLimit = None
        self.position_target = None
        self._init_scaling()

    def _init_scaling(self):
        """Initialize the scaling of the target"""
        if self._type_target == "fixed":
            self._upperPositionLimit = self._range_target + np.ones(3)
            self._lowerPositionLimit = self._range_target - np.ones(3)
        elif self._type_target == "box":
            self._upperPositionLimit = self._range_target[:3] + self._range_target[3:6]
            self._lowerPositionLimit = self._range_target[:3] - self._range_target[6:]
        elif self._type_target == "sphere":
            self._upperPositionLimit = self._range_target[:3] + self._range_target[
                3
            ] * np.ones(3)
            self._lowerPositionLimit = self._range_target[:3] - self._range_target[
                3
            ] * np.ones(3)

    def create_sphere(self):
        """Create a sphere target position for the environment"""
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

    def create_box(self):
        """Create a box target position for the environment"""
        size_low = self._range_target[3:6]
        size_high = self._range_target[6:9]

        return np.array(
            [
                self._range_target[0] + np.random.uniform(size_low[0], size_high[0]),
                self._range_target[1] + np.random.uniform(size_low[1], size_high[1]),
                self._range_target[2] + np.random.uniform(size_low[2], size_high[2]),
            ],
        )

    def create_target(self):
        """Create a target position for the environment"""

        if self._type_target == "fixed":
            self.position_target = self.range
        elif self._type_target == "box":
            self.position_target = self.create_box()
        elif self._type_target == "sphere":
            self.position_target = self.create_sphere()
        else:
            msg = "Unknown target type"
            raise ValueError(msg)

    def _sort_datas(self, params_env):
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
                    np.array([params_env["targetPosition"]]),
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
