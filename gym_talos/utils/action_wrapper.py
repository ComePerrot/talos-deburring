import numpy as np


class action_wrapper:
    def __init__(
        self,
        rl_controlled_IDs,
        rmodel,
        scaling_factor=1,
        scaling_mode="full_range",
        initial_pose=None,
    ):
        self.rl_controlled_IDs = rl_controlled_IDs
        self.rmodel = rmodel
        self.scaling_factor = scaling_factor
        self.scaling_mode = scaling_mode
        self.q0 = initial_pose
        self._init_actScaler()
        if scaling_mode == "differential":
            # Check scaling
            self._check_scaling()

    def _init_actScaler(self):
        """Initializes the action scaler using robot model limits"""
        self.lowerActLim = np.array(
            [
                self.rmodel.lowerPositionLimit[joint_ID]
                for joint_ID in self.rl_controlled_IDs
            ],
        )
        self.upperActLim = np.array(
            [
                self.rmodel.upperPositionLimit[joint_ID]
                for joint_ID in self.rl_controlled_IDs
            ],
        )

        self.avgAct = (self.upperActLim + self.lowerActLim) / 2
        self.diffAct = (self.upperActLim - self.lowerActLim) / 2

    def _check_scaling(self):
        """Checks range of scaled action

        Checks that the scaled action stays inside the kinematic limits of the
        robot (only used when the scaling mode is differential)
        """
        upper_action = self.q0 + self.diffAct * self.scaling_factor
        lower_action = self.q0 - self.diffAct * self.scaling_factor
        if (upper_action > self.upperActLim).any() or (
            lower_action < self.lowerActLim
        ).any():
            msg = "Scaling of action is not inside of the model limits"
            raise ValueError(msg)

    def action(self, action):
        """Scale the action given by the agent based on the chosen scaling mode

        Args:
            action: normalized action given by the agent in the range [-1, 1]

        Returns:
            unnormalized reference based on the chosen scaling mode
        """
        scaled_action = action * self.diffAct * self.scaling_factor
        if self.scaling_mode == "full_range":
            reference = self.avgAct + scaled_action
        elif self.scaling_mode == "differential" and self.q0 is not None:
            reference = self.q0 + scaled_action
        else:
            msg = "Invalid scaling mode or missing initial pose."
            raise ValueError(msg)

        return reference
