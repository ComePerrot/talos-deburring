import numpy as np
import warnings


class ActionWrapper:
    def __init__(
        self,
        rmodel,
        rl_controlled_joints,
        initial_state,
        scaling_factor=1,
        scaling_mode="full_range",
        clip_action=False,
    ):
        # Check input arguments
        for joint in rl_controlled_joints:
            assert joint in rmodel.names.tolist(), f"{joint} not in robot model."
        assert (
            len(initial_state) == rmodel.nq + rmodel.nv
        ), f"Size mismatch: initial state is of size ({len(initial_state)}) instead of ({rmodel.nq + rmodel.nv})."
        assert scaling_mode in [
            "full_range",
            "differential",
        ], f"Unexpected '{scaling_mode}' scaling_mode, must be either 'full_range' or 'differential'."
        assert isinstance(clip_action, bool), "clip_action must be a boolean."

        self.rmodel = rmodel
        self.rl_controlled_joints = rl_controlled_joints
        self.rl_controlled_ids = np.array(
            [
                self.rmodel.names.tolist().index(joint_name) - 2 + 7
                for joint_name in self.rl_controlled_joints
            ],
        )

        # initial state
        self.x0 = initial_state.copy()
        self.partial_x0 = [self.x0[i] for i in self.rl_controlled_ids]

        self.reference_state = self.x0.copy()

        self.scaling_factor = scaling_factor
        self.scaling_mode = scaling_mode
        self.clip_action = clip_action

        self._compute_action_data()

        if scaling_mode == "differential" and not clip_action:
            self._check_scaling()

    def _compute_action_data(self):
        self.lower_action_limit = np.array(
            [
                self.rmodel.lowerPositionLimit[joint_ID]
                for joint_ID in self.rl_controlled_ids
            ],
        )
        self.upper_action_limit = np.array(
            [
                self.rmodel.upperPositionLimit[joint_ID]
                for joint_ID in self.rl_controlled_ids
            ],
        )

        self.action_average = (self.upper_action_limit + self.lower_action_limit) / 2
        self.action_amplitude = (self.upper_action_limit - self.lower_action_limit) / 2

    def _check_scaling(self):
        """Checks range of scaled action

        Checks that the scaled action stays inside the kinematic limits of the
        robot (only used when the scaling mode is differential)
        """
        upper_action = self.compute_partial_state(np.ones(len(self.rl_controlled_ids)))
        lower_action = self.compute_partial_state(-np.ones(len(self.rl_controlled_ids)))
        if (upper_action > self.upper_action_limit).any() or (
            lower_action < self.lower_action_limit
        ).any():
            msg = "Scaling of action is not inside of the model limits."
            print(msg)
            # warnings.warn(msg, stacklevel=2)

    def compute_reference_state(self, action):
        partial_state = self.compute_partial_state(action)

        for i, state_id in enumerate(self.rl_controlled_ids):
            self.reference_state[state_id] = partial_state[i]

        return self.reference_state

    def compute_partial_state(self, action):
        scaled_action = self._scale_action(action)
        if self.scaling_mode == "full_range":
            partial_reference = self.action_average + scaled_action
        else:
            partial_reference = self.partial_x0 + scaled_action

        if self.clip_action:
            partial_reference = self._clip_reference(partial_reference)

        return partial_reference

    def _scale_action(self, action):
        return self.scaling_factor * np.array(action) * self.action_amplitude

    def _clip_reference(self, partial_reference):
        return [
            max(
                min(partial_reference[i], self.upper_action_limit[i]),
                self.lower_action_limit[i],
            )
            for i in range(len(partial_reference))
        ]

    def update_initial_state(self, state):
        self.x0 = self.reference_state = state.copy()
        self.x0[self.rmodel.nq :] = np.zeros(self.rmodel.nv)

        self.partial_x0 = [self.x0[i] for i in self.rl_controlled_ids].copy()
