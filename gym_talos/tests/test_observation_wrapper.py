import numpy as np
import pinocchio as pin
import unittest

from gym_talos.tests.factory import RobotModelFactory
from gym_talos.utils.observation_wrapper import observation_wrapper


class ObservationTestCase(unittest.TestCase):
    def setUp(self):
        model_factory = RobotModelFactory()
        self.rmodel = model_factory.get_rmodel()
        self.upper_limit = self.rmodel.upperPositionLimit
        self.upper_limit[:7] = np.ones(7) * 5
        self.lower_limit = self.rmodel.lowerPositionLimit
        self.lower_limit[:7] = np.zeros(7) * 5
        self.target_handler = model_factory.get_target_handler()
        normalize_obs = True
        self.history_size = 5
        self.prediction_size = 3

        self.observation_wrapper = observation_wrapper(
            normalize_obs,
            self.rmodel,
            self.target_handler,
            self.history_size,
            self.prediction_size,
        )

        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.x0 = np.concatenate([q0, np.zeros(self.rmodel.nv)])

    def generate_posture(self):
        return np.concatenate(
            [
                pin.randomConfiguration(
                    self.rmodel,
                    self.lower_limit,
                    self.upper_limit,
                ),
                np.zeros(self.rmodel.nv),
            ],
        )

    def test_reset(self):
        x_measured = self.generate_posture()
        target_position = self.target_handler.generate_target()[0]
        x_future_list = [self.generate_posture() for _ in range(self.prediction_size)]
        observation = self.observation_wrapper.reset(
            x_measured,
            target_position,
            x_future_list,
        )
        self.assertEqual(
            observation.shape, (self.observation_wrapper.observation_size,)
        )

    def test_get_observation(self):
        x_measured = self.generate_posture()
        target_position = self.target_handler.generate_target()[0]
        x_future_list = [self.generate_posture() for _ in range(self.prediction_size)]
        self.observation_wrapper.reset(x_measured, target_position, x_future_list)
        observation = self.observation_wrapper.get_observation(
            x_measured, x_future_list
        )
        self.assertEqual(
            observation.shape,
            (self.observation_wrapper.observation_size,),
        )

    def test_normalize_state(self):
        for _ in range(100):
            normalized_state = self.observation_wrapper.normalize_state(
                self.generate_posture(),
            )
            self.assertLessEqual(np.abs(normalized_state).max(), 1)

    def test_normalize_target(self):
        targets = self.target_handler.generate_target(100)
        for target in targets:
            normalized_target = self.observation_wrapper.normalize_target(target)
            self.assertLessEqual(np.abs(normalized_target).max(), 1)


if __name__ == "__main__":
    unittest.main()
