from stable_baselines3 import SAC

from gym_talos.utils.action_wrapper import action_wrapper
from gym_talos.utils.observation_wrapper import observation_wrapper

class RLPostureController:
    def __init__(self, model_path, kwargs_action, kwargs_observation):
        self.model = SAC.load(model_path, env=None)
        self.action_wrapper = action_wrapper(kwargs_action)
        self.observation_wrapper = observation_wrapper(kwargs_observation)

    def step(self, x_measured, x_future_list):
        observation = self.observation_wrapper.get_observation(x_measured,x_future_list)
        action, _ = self.model.predict(observation, deterministic=True)
        posture = self.action_wrapper.action(action)

        return(posture)