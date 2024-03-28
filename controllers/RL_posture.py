from stable_baselines3 import SAC

from gym_talos.utils.action_wrapper import ActionWrapper
from gym_talos.utils.observation_wrapper import observation_wrapper


class RLPostureController:
    def __init__(self, model_path, x0, kwargs_action, kwargs_observation):
        self.x0 = x0
        self.model = SAC.load(model_path, env=None)
        self.action_wrapper = ActionWrapper(**kwargs_action)
        self.observation_wrapper = observation_wrapper(**kwargs_observation)

    def step(self, x_measured, x_future_list):
        observation = self.observation_wrapper.get_observation(
            x_measured,
            x_future_list,
        )
        action, _ = self.model.predict(observation, deterministic=True)
        self.action_wrapper.update_initial_state(x_measured)
        x_reference = self.action_wrapper.compute_reference_state(action)

        return x_reference  # noqa: RET504
