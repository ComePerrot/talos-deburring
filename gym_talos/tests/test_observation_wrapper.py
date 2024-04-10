import numpy as np
import unittest

from gym_talos.tests.factory import RobotModelFactory
from gym_talos.utils.observation_wrapper import observation_wrapper


class ObservationTestCase(unittest.TestCase):
    def setUp(self):
        model_factory = RobotModelFactory()
        self.rmodel = model_factory.get_rmodel()
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

    def test_reset(self):
        x_measured = np.random.rand(self.rmodel.nq + self.rmodel.nv)
        target_position = np.random.rand(3)
        x_future_list = [np.random.rand(self.rmodel.nq + self.rmodel.nv) for _ in range(self.prediction_size)]
        observation = self.observation_wrapper.reset(x_measured, target_position, x_future_list)
        self.assertEqual(observation.shape, (self.observation_wrapper.observation_size,))

    def test_get_observation(self):
        x_measured = np.random.rand(self.rmodel.nq + self.rmodel.nv)
        target_position = np.random.rand(3)
        x_future_list = [np.random.rand(self.rmodel.nq + self.rmodel.nv) for _ in range(self.prediction_size)]
        self.observation_wrapper.reset(x_measured, target_position, x_future_list)
        observation = self.observation_wrapper.get_observation(x_measured, x_future_list)
        self.assertEqual(observation.shape, (self.observation_wrapper.observation_size,))

    def test_normalize_state(self):
        normalized_state = self.observation_wrapper.normalize_state(self.x0)
        self.assertLessEqual(np.abs(normalized_state).max(), 1)

    def test_normalize_target(self):
        target = self.target_handler.generate_target()
        normalized_target = self.observation_wrapper.normalize_target(target)
        self.assertLessEqual(np.abs(normalized_target).max(), 1)


if __name__ == "__main__":
    unittest.main()
