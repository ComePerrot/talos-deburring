import numpy as np
import pinocchio as pin
import unittest

from robot_description.path_getter import urdf_path, srdf_path

from simulator.bullet_Talos import TalosDeburringSimulator


class ModelFactory:
    @staticmethod
    def create_robot_model():
        # Create and return a complex object
        rmodel = pin.buildModelFromUrdf(
            urdf_path["example_robot_data"],
            pin.JointModelFreeFlyer(),
        )
        pin.loadReferenceConfigurations(rmodel, srdf_path, False)
        return rmodel


class TestBulletSimulator(unittest.TestCase):
    def setUp(self):
        self.factory = ModelFactory()
        self.rmodel = self.factory.create_robot_model()
        self.nq = self.rmodel.nq

        self.controlled_joints_ids = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            30,
            31,
            32,
            33,
        ]

        self.nq = len(self.controlled_joints_ids)

        self.simulator = TalosDeburringSimulator(
            urdf_path["custom"],
            self.rmodel,
            self.controlled_joints_ids,
            randomInit=False,
            enableGUI=False,
            enableGravity=True,
            dt=1e-3,
            cutoff_frequency=0,
        )

    def test_reset(self):
        nb_tries = 10
        torques = np.array(
            [
                -0.41736194,
                2.38640311,
                1.67973602,
                -49.08087835,
                2.32353507,
                -2.00753922,
                -0.37375559,
                -3.44156605,
                1.80754766,
                -49.13471236,
                2.25634006,
                1.12429376,
                0.74195141,
                -4.01560375,
                -2.15259184,
                -0.97061155,
                3.67786258,
                -10.08465081,
                3.08987902,
                1.05611906,
                -2.00361954,
                -9.38055396,
            ],
        )
        x0_list = np.zeros((nb_tries, len(self.simulator.getRobotState())))
        x1_list = np.zeros((nb_tries, len(self.simulator.getRobotState())))
        for i in range(nb_tries):
            self.simulator.reset([0, 0, 0])
            x0_list[i, :] = self.simulator.getRobotState().copy()
            self.simulator.step(torques)
            x1_list[i, :] = self.simulator.getRobotState().copy()
            for _ in range(100):
                self.simulator.step(torques)

        for x0 in x0_list:
            for x0_i, x0_i_ref in zip(x0, x0_list[0]):
                self.assertAlmostEqual(x0_i, x0_i_ref, 2)

        for x1 in x1_list:
            for x1_i, x1_i_ref in zip(x1, x1_list[0]):
                self.assertAlmostEqual(x1_i, x1_i_ref, 2)


if __name__ == "__main__":
    unittest.main()
