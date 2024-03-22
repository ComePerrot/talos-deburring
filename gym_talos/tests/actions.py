import numpy as np
import unittest

from gym_talos.tests.factory import RobotModelFactory
from gym_talos.utils.action_wrapper import ActionWrapper


class ActionTestCase(unittest.TestCase):
    def setUp(self):
        model_factory = RobotModelFactory()
        rmodel = model_factory.get_rmodel()
        q0 = rmodel.referenceConfigurations["half_sitting"]
        self.x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
        rl_controlled_joints = [
            "arm_left_1_joint",
            "arm_left_2_joint",
            "arm_left_3_joint",
            "arm_left_4_joint",
        ]
        self.rl_controlled_ids = np.array(
            [
                rmodel.names.tolist().index(joint_name) - 2 + 7
                for joint_name in rl_controlled_joints
            ],
        )

        self.action_wrapper = ActionWrapper(
            rmodel,
            rl_controlled_joints,
            self.x0,
            scaling_factor=1,
            scaling_mode="full_range",
            clip_action=False,
        )

    def test_compute_reference_state(self):
        state_ref_neutral = self.action_wrapper.compute_reference_state([0, 0, 0, 0])
        x0_neutral = self.x0.copy()
        for i, i_joint in enumerate(self.rl_controlled_ids):
            x0_neutral[i_joint] = self.action_wrapper.action_average[i]
        np.testing.assert_allclose(state_ref_neutral, x0_neutral)

        state_ref_upper = self.action_wrapper.compute_reference_state([1, 1, 1, 1])
        x0_upper = self.x0
        for i in self.rl_controlled_ids:
            x0_upper[i] = self.action_wrapper.rmodel.upperPositionLimit[i]
        np.testing.assert_allclose(state_ref_upper, x0_upper)

        state_ref_lower = self.action_wrapper.compute_reference_state([-1, -1, -1, -1])
        x0_lower = self.x0
        for i in self.rl_controlled_ids:
            x0_lower[i] = self.action_wrapper.rmodel.lowerPositionLimit[i]
        np.testing.assert_allclose(state_ref_lower, x0_lower)

    def test_compute_partial_state(self):
        partial_state_neutral = self.action_wrapper.compute_partial_state([0, 0, 0, 0])
        partial_x0_neutral = self.action_wrapper.action_average
        np.testing.assert_allclose(partial_state_neutral, partial_x0_neutral)

        partial_state_upper = self.action_wrapper.compute_partial_state([1, 1, 1, 1])
        partial_x0_upper = self.action_wrapper.upper_action_limit
        np.testing.assert_allclose(partial_state_upper, partial_x0_upper)

        partial_state_lower = self.action_wrapper.compute_partial_state(
            [-1, -1, -1, -1],
        )
        partial_x0_lower = self.action_wrapper.lower_action_limit
        np.testing.assert_allclose(partial_state_lower, partial_x0_lower)


if __name__ == "__main__":
    unittest.main()
