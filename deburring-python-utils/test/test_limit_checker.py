import numpy as np
import pinocchio as pin
import unittest

from robot_description.path_getter import urdf_path, srdf_path

from limit_checker_talos.limit_checker import LimitChecker


class ModelFactory:
    @staticmethod
    def create_robot_model():
        # Create and return a complex object
        rmodel = pin.buildModelFromUrdf(urdf_path["example_robot_data"])
        pin.loadReferenceConfigurations(rmodel, srdf_path, False)
        return rmodel


class TestLimitChecker(unittest.TestCase):
    def setUp(self):
        self.factory = ModelFactory()
        self.rmodel = self.factory.create_robot_model()
        self.nq = self.rmodel.nq

        self.limit_checker = LimitChecker(self.rmodel)

    def test_position_limit(self):
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        q_upper = self.rmodel.upperPositionLimit + 0.1
        q_lower = self.rmodel.lowerPositionLimit - 0.1

        for i, q_i in enumerate(q_upper):
            self.assertTupleEqual(
                self.limit_checker.check_position_limit(q_i, i),
                (
                    self.rmodel.names[i + 1],
                    (q_i, self.limit_checker.upper_pos_limits[i]),
                ),
            )

        for i, q_i in enumerate(q_lower):
            self.assertTupleEqual(
                self.limit_checker.check_position_limit(q_i, i),
                (
                    self.rmodel.names[i + 1],
                    (q_i, self.limit_checker.lower_pos_limits[i]),
                ),
            )

        for i, q_i in enumerate(q0):
            self.assertIsNone(self.limit_checker.check_position_limit(q_i, i))

    def test_velocity_limit(self):
        v0 = np.zeros(self.nq)
        v_error = self.rmodel.velocityLimit + 0.1
        for i, v_i in enumerate(v_error):
            self.assertTupleEqual(
                self.limit_checker.check_velocity_limit(v_i, i),
                (
                    self.rmodel.names[i + 1],
                    (v_i, self.limit_checker.velocity_limits[i]),
                ),
            )

            self.assertTupleEqual(
                self.limit_checker.check_velocity_limit(-v_i, i),
                (
                    self.rmodel.names[i + 1],
                    (-v_i, self.limit_checker.velocity_limits[i]),
                ),
            )

        for i, v_i in enumerate(v0):
            self.assertIsNone(self.limit_checker.check_velocity_limit(v_i, i))

    def test_effort_limit(self):
        u0 = np.zeros(self.nq)
        u_error = self.rmodel.effortLimit + 0.1
        for i, u_i in enumerate(u_error):
            self.assertTupleEqual(
                self.limit_checker.check_effort_limit(u_i, i),
                (
                    self.rmodel.names[i + 1],
                    (u_i, self.limit_checker.effort_limits[i]),
                ),
            )

            self.assertTupleEqual(
                self.limit_checker.check_effort_limit(-u_i, i),
                (
                    self.rmodel.names[i + 1],
                    (-u_i, self.limit_checker.effort_limits[i]),
                ),
            )

        for i, u_i in enumerate(u0):
            self.assertIsNone(self.limit_checker.check_effort_limit(u_i, i))

    def test_all_limits(self):
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        v0 = np.zeros(self.nq)
        u0 = np.zeros(self.nq)
        q_error = np.full(self.nq, -10)
        v_error = np.full(self.nq, -10)
        u_error = np.full(self.nq, -1000)

        self.limit_checker.verbose = False
        self.assertFalse(self.limit_checker.are_limits_broken(q0, v0, u0))
        self.assertTrue(self.limit_checker.are_limits_broken(q_error, v_error, u_error))

        self.limit_checker.verbose = True
        self.assertFalse(self.limit_checker.are_limits_broken(q0, v0, u0))
        self.assertListEqual(
            ["position", "velocity", "effort"],
            list(
                self.limit_checker.are_limits_broken(q_error, v_error, u_error).keys(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
