import numpy as np
from stable_baselines3 import SAC


class RLTorqueController:
    def __init__(self, model_path, kwargs_action, kwargs_observation):
        self.model = SAC.load(model_path, env=None)
        self.action_wrapper = action_wrapper(kwargs_action)
        self.observation_wrapper = observation_wrapper(kwargs_observation)

    def step(self, x_measured):
        observation = self.observation_wrapper.get_observation(x_measured)
        action, _ = self.model.predict(observation, deterministic=True)
        torque = self.action_wrapper.action(action)

        return torque  # noqa: RET504


class observation_wrapper:
    def __init__(self):
        self.lowerObsLim = np.concatenate(
            (
                self.rmodel.lowerPositionLimit,
                -self.rmodel.velocityLimit,
            ),
        )

        self.upperObsLim = np.concatenate(
            (
                self.rmodel.upperPositionLimit,
                self.rmodel.velocityLimit,
            ),
        )

        self.avgObs = (self.upperObsLim + self.lowerObsLim) / 2
        self.diffObs = self.upperObsLim - self.lowerObsLim

        self.lowerGoalLim = -3 * np.ones(3)
        self.upperGoalLim = 3 * np.ones(3)
        self.avgGoal = (self.upperGoalLim + self.lowerGoalLim) / 2
        self.diffGoal = self.upperGoalLim - self.lowerGoalLim

        self.lowerGoalLim = -3 * np.ones(3)
        self.upperGoalLim = 3 * np.ones(3)
        self.avgGoal = (self.upperGoalLim + self.lowerGoalLim) / 2
        self.diffGoal = self.upperGoalLim - self.lowerGoalLim

    def get_observation(self, x_measured):
        pass
        # return (goal - self.avgGoal) / self.diffGoal
        # return (x_measured - self.avgObs) / self.diffObs
        # return (target - self.avgGoal) / self.diffGoal

        # final_obs.spaces["observation"] = np.array(observation)
        # final_obs.spaces["achieved_goal"] = np.array(achieved_goal)
        # final_obs.spaces["desired_goal"] = np.array(desired_goal)
        # return collections.OrderedDict(final_obs)


class action_wrapper:
    def __init__(self):
        self.torqueScale = self.torqueScaleCoeff * np.array(self.rmodel.effortLimit)

    def get_action(self, action):
        return (
            self.torqueScale[6:] * action[7:]
            if self.simulator.has_free_flyer
            else self.torqueScale * action
        )
