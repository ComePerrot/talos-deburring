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
        )

    def test_upper_bound(self):
        upper_limit = np.array(
            [
                self.action_wrapper.rmodel.upperPositionLimit[joint_ID]
                for joint_ID in self.action_wrapper.rl_controlled_IDs
            ],
        )
        scaled_action = self.action_wrapper.action(np.ones(4))

        self.assertEqual(scaled_action, upper_limit)


if __name__ == "__main__":
    unittest.main()
