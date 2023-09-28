import numpy as np
import unittest

from gym_talos.tests.factory import RobotModelFactory
from gym_talos.utils.action_wrapper import action_wrapper


class ActionTestCase(unittest.TestCase):
    def setUp(self):
        model_factory = RobotModelFactory()
        rmodel = model_factory.get_rmodel()
        rl_controlled_joints = [
            "arm_left_1_joint",
            "arm_left_2_joint",
            "arm_left_3_joint",
            "arm_left_4_joint",
        ]
        rl_controlled_IDs = np.array(
            [
                rmodel.names.tolist().index(joint_name) - 2 + 7
                for joint_name in rl_controlled_joints
            ],
        )
        self.action_wrapper = action_wrapper(
            rl_controlled_IDs,
            rmodel,
            scaling_factor=1,
            scaling_mode="full_range",
            initial_pose=None,
        )

    def test_upper_bound(self):
        upper_limit = np.array(
            [
                self.action_wrapper.rmodel.upperPositionLimit[joint_ID]
                for joint_ID in self.action_wrapper.rl_controlled_IDs
            ],
        )
        scaled_action = self.action_wrapper.action(
            np.ones(len(self.action_wrapper.rl_controlled_IDs)),
        )

        for i in range(len(scaled_action)):
            self.assertAlmostEqual(scaled_action[i], upper_limit[i], 8)

    def test_lower_bound(self):
        lower_limit = np.array(
            [
                self.action_wrapper.rmodel.lowerPositionLimit[joint_ID]
                for joint_ID in self.action_wrapper.rl_controlled_IDs
            ],
        )
        scaled_action = self.action_wrapper.action(
            -np.ones(len(self.action_wrapper.rl_controlled_IDs)),
        )

        for i in range(len(scaled_action)):
            self.assertAlmostEqual(scaled_action[i], lower_limit[i], 8)


if __name__ == "__main__":
    unittest.main()
