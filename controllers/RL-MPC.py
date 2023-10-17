import numpy as np
import pinocchio as pin

from stable_baselines3 import SAC
from gym_talos.utils.action_wrapper import action_wrapper
from gym_talos.utils.observation_wrapper import observation_wrapper

from deburring_mpc import OCP

class HybridController:
    def __init__(self, model_path, kwargs_action, kwargs_observation,
                 x_initial, target_pos, params_designer, param_ocp):
        self._load_RL_agent(model_path, kwargs_action, kwargs_observation)

        self._load_MPC(x_initial, target_pos, params_designer, param_ocp)

    def _load_RL_agent(self, model_path, kwargs_action, kwargs_observation):
        self.model = SAC.load(model_path, env=None)
        self.action_wrapper = action_wrapper(kwargs_action)
        self.observation_wrapper = observation_wrapper(kwargs_observation)

    def _load_MPC(self, pinWrapper, x_initial, target_pos, param_ocp):
        self.time = 0
        self.num_control_knots = 10

        self.pinWrapper = pinWrapper

        self.param_ocp["state_weights"] = np.array(self.param_ocp["state_weights"])
        self.param_ocp["control_weights"] = np.array(self.param_ocp["control_weights"])

        self.oMtarget = pin.SE3.Identity()
        self.oMtarget.translation[0] = target_pos[0]
        self.oMtarget.translation[1] = target_pos[1]
        self.oMtarget.translation[2] = target_pos[2]

        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.crocoWrapper = OCP(param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(x_initial, self.oMtarget)

        self.x0 = x_initial
        self.q0 = self.pinWrapper.get_x0().copy()

    def step(self, x_measured, x_future_list):
        self.timer += 1
        
        observation = self.observation_wrapper.get_observation(x_measured,x_future_list)
        action, _ = self.model.predict(observation, deterministic=True)
        posture = self.action_wrapper.action(action)

        arm_reference = self.action_handler.action(posture)

        posture_reference = self.q0
        for i in range(self.action_dimension):
            posture_reference[self.rl_controlled_IDs[i]] = arm_reference[i]

        torque_sum = 0

        for _ in range(self.numOCPSteps):
            x0 = self.simulator.getRobotState()
            oMtool = self.pinWrapper.get_end_effector_frame()

            for _ in range(self.numSimulationSteps):
                x_measured = self.simulator.getRobotState()
                torques = (
                    self.crocoWrapper.torque
                    + self.crocoWrapper.gain
                    @ self.crocoWrapper.state.diff(x_measured, x0)
                )

                self.simulator.step(torques, oMtool)

            torque_sum += np.linalg.norm(torques)

            x_measured = self.simulator.getRobotState()

            self.pinWrapper.update_reduced_model(x_measured)

            self.crocoWrapper.recede()
            self.crocoWrapper.change_goal_cost_activation(self.horizon_length - 1, True)
            self.crocoWrapper.change_posture_reference(
                self.horizon_length - 1,
                posture_reference,
            )
            self.crocoWrapper.change_posture_reference(
                self.horizon_length,
                posture_reference,
            )

            self.crocoWrapper.solve(x_measured)

        avg_torque_norm = torque_sum / (self.numOCPSteps * self.numSimulationSteps)

        observation = self.observation_handler.get_observation(
            x_measured,
            self.crocoWrapper.solver.xs,
        )
        terminated = self._checkTermination(x_measured)
        truncated = self._checkTruncation(x_measured)
        reward = self._getReward(avg_torque_norm, x_measured, terminated, truncated)

        infos = {"dst": self.distance_tool_target, "time": self.reach_time}

        return observation, reward, terminated, truncated, infos

        return(torques)

