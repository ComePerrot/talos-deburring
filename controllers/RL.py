from stable_baselines3 import SAC

class rl_wrapper:
    def __init__(self, model_path):
        self.model = SAC.load(model_path, env=None)

    def step(self, x_measured):
        observation = x_measured
        action, _ = self.model.predict(observation, deterministic=True)
        torques = action

        return(torques)

