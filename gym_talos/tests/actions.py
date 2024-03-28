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
        q1 = q0 + 0.1
        self.x1 = np.concatenate([q1, np.zeros(rmodel.nv)])
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
        self.na = len(rl_controlled_joints)

        self.action_wrapper = ActionWrapper(
            rmodel,
            rl_controlled_joints,
            self.x0,
            scaling_factor=1,
            scaling_mode="full_range",
            clip_action=True,
        )

    def assert_state_equal(self, actual_reference, expected_reference):
        np.testing.assert_allclose(actual_reference, expected_reference)

    def assert_state_less_equal(self, actual_reference, expected_reference):
        for state_ref_i, state_expected_i in zip(actual_reference, expected_reference):
            self.assertLessEqual(state_ref_i, state_expected_i)

    def assert_state_more_equal(self, actual_reference, expected_reference):
        for state_ref_i, state_expected_i in zip(actual_reference, expected_reference):
            self.assertLessEqual(state_expected_i, state_ref_i)

    def test_compute_reference_state(self):
        for i, i_joint in enumerate(self.rl_controlled_ids):
            self.x0[i_joint] = self.action_wrapper.action_average[i]
        self.assert_state_equal(
            self.action_wrapper.compute_reference_state(np.zeros(self.na)),
            self.x0,
        )

        self.x0[self.rl_controlled_ids] = self.action_wrapper.rmodel.upperPositionLimit[
            self.rl_controlled_ids
        ]
        self.assert_state_equal(
            self.action_wrapper.compute_reference_state(np.ones(self.na)),
            self.x0,
        )

        self.x0[self.rl_controlled_ids] = self.action_wrapper.rmodel.lowerPositionLimit[
            self.rl_controlled_ids
        ]
        self.assert_state_equal(
            self.action_wrapper.compute_reference_state(-np.ones(self.na)),
            self.x0,
        )

    def test_compute_reference_state_differential(self):
        self.action_wrapper.scaling_mode = "differential"
        self.assert_state_equal(
            self.action_wrapper.compute_reference_state(np.zeros(self.na)),
            self.x0,
        )

        self.x0[self.rl_controlled_ids] = self.action_wrapper.rmodel.upperPositionLimit[
            self.rl_controlled_ids
        ]
        self.assert_state_less_equal(
            self.action_wrapper.compute_reference_state(np.ones(self.na)),
            self.x0,
        )

        self.x0[self.rl_controlled_ids] = self.action_wrapper.rmodel.lowerPositionLimit[
            self.rl_controlled_ids
        ]
        self.assert_state_more_equal(
            self.action_wrapper.compute_reference_state(-np.ones(self.na)),
            self.x0,
        )

    def test_compute_partial_state(self):
        self.assert_state_equal(
            self.action_wrapper.compute_partial_state(np.zeros(self.na)),
            self.action_wrapper.action_average,
        )
        self.assert_state_equal(
            self.action_wrapper.compute_partial_state(np.ones(self.na)),
            self.action_wrapper.upper_action_limit,
        )
        self.assert_state_equal(
            self.action_wrapper.compute_partial_state(-np.ones(self.na)),
            self.action_wrapper.lower_action_limit,
        )

    def test_compute_partial_state_differential(self):
        self.action_wrapper.scaling_mode = "differential"
        self.assert_state_equal(
            self.action_wrapper.compute_partial_state(np.zeros(self.na)),
            [self.x0[i] for i in self.rl_controlled_ids],
        )
        self.assert_state_less_equal(
            self.action_wrapper.compute_partial_state(np.ones(self.na)),
            self.action_wrapper.upper_action_limit,
        )
        self.assert_state_more_equal(
            self.action_wrapper.compute_partial_state(-np.ones(self.na)),
            self.action_wrapper.lower_action_limit,
        )

    def test_update_initial_state(self):
        self.action_wrapper.update_initial_state(self.x1)

        self.test_compute_partial_state()

        self.x0 = self.x1
        self.test_compute_partial_state_differential()


if __name__ == "__main__":
    unittest.main()
