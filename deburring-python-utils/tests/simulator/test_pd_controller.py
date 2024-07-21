import unittest
from simulator.pd_controller import PDController


class TestPDController(unittest.TestCase):

    def setUp(self):
        self.controller = PDController()

    def test_compute_control_torso(self):
        joint_name = "torso_1_joint"
        measured_pos = 0.1
        measured_vel = 0.2
        expected_torque = -500.0 * measured_pos - 20.0 * measured_vel
        actual_torque = self.controller.compute_control(
            joint_name,
            measured_pos,
            measured_vel,
        )
        self.assertEqual(expected_torque, actual_torque)

    def test_compute_control_arm(self):
        joint_name = "arm_left_1_joint"
        measured_pos = 0.1
        measured_vel = 0.2
        expected_torque = 0.06 - 100.0 * (measured_pos - 0.4) - 8.0 * measured_vel
        actual_torque = self.controller.compute_control(
            joint_name,
            measured_pos,
            measured_vel,
        )
        self.assertEqual(expected_torque, actual_torque)

    def test_compute_control_leg(self):
        joint_name = "leg_left_1_joint"
        measured_pos = 0.1
        measured_vel = 0.2
        expected_torque = -800.0 * measured_pos - 35.0 * measured_vel
        actual_torque = self.controller.compute_control(
            joint_name,
            measured_pos,
            measured_vel,
        )
        self.assertEqual(expected_torque, actual_torque)


if __name__ == "__main__":
    unittest.main()
