import gymnasium as gym
import numpy as np

from deburring_mpc import RobotDesigner
from gym_talos.utils.action_wrapper import ActionWrapper
from gym_talos.utils.create_target import TargetGoal
from gym_talos.utils.observation_wrapper import observation_wrapper
from limit_checker_talos.limit_checker import LimitChecker
from robot_description.path_getter import srdf_path, urdf_path
from simulator.bullet_Talos import TalosDeburringSimulator


class EnvTalosPosition(gym.Env):
    def __init__(self, params_robot, params_env, GUI=False):
        params_designer = params_robot["designer"]

        self._init_parameters(params_robot, params_env)

        # Robot model handler
        self.pinWrapper = RobotDesigner()
        params_designer["end_effector_position"] = np.array(
            params_designer["end_effector_position"],
        )
        params_designer["urdf_path"] = urdf_path[params_designer["urdf_type"]]
        params_designer["srdf_path"] = srdf_path
        self.pinWrapper.initialize(params_designer)

        self.limit_checker = LimitChecker(
            self.pinWrapper.get_rmodel(),
            extra_limits=params_env["extra_limits"],
            verbose=False,
        )

        # Simulator
        self.simulator = TalosDeburringSimulator(
            URDF=params_designer["urdf_path"],
            rmodel_complete=self.pinWrapper.get_rmodel_complete(),
            controlled_joints_ids=self.pinWrapper.get_controlled_joints_ids(),
            enable_GUI=GUI,
            dt=params_env["time_step_simulation"],
        )

        # PD controller
        self.pd_controller = PDController(
            self.pinWrapper.get_rmodel().names[2:],
            params_robot["pd_controller"]["gains"],
        )

        # Target handler
        self.target_handler = TargetGoal(params_env["target"])
        self.target_handler.create_target()

        # Observation handler
        self.observation_handler = observation_wrapper(
            self.normalize_obs,
            self.pinWrapper.get_rmodel(),
            self.target_handler,
            params_env["observation"]["history_obs"],
            0,
        )

        # Action handler
        self.action_handler = ActionWrapper(
            self.pinWrapper.get_rmodel(),
            self.pinWrapper.get_rmodel().names[2:],
            initial_state=self.pinWrapper.get_x0().copy(),
            scaling_factor=params_env["action"]["action_scale"],
            scaling_mode=params_env["action"]["action_type"],
            clip_action=params_env["action"]["clip_action"],
        )

        self._init_env_variables(
            action_dimension=self.n_joints,
            observation_dimension=self.observation_handler.observation_size,
        )

    def _init_parameters(self, params_robot, params_env):
        self.n_joints = len(params_robot["designer"]["controlled_joints_names"]) - 1
        self.num_sim_steps = int(
            params_robot["pd_controller"]["time_step_controller"]
            / params_env["time_step_simulation"],
        )
        self.max_step = int(
            params_env["max_time"]
            / (self.num_sim_steps * params_env["time_step_simulation"]),
        )
        self.min_height = params_env["min_height"]

        self.normalize_obs = params_env["observation"]["normalize_obs"]

        #  Reward parameters
        self.distance_threshold = params_env["reward"]["distance_threshold"]
        self.weight_success = params_env["reward"]["w_success"]
        self.weight_distance = params_env["reward"]["w_distance"]
        self.weight_truncation = params_env["reward"]["w_penalization_truncation"]
        self.weight_energy = params_env["reward"]["w_penalization_torque"]

    def _init_env_variables(self, action_dimension, observation_dimension):
        self.timer = 0

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(action_dimension,),
            dtype=np.float32,
        )

        if self.normalize_obs:
            self.observation_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(observation_dimension,),
                dtype=np.float64,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-5,
                high=5,
                shape=(observation_dimension,),
                dtype=np.float64,
            )

        self.distance_tool_target = None
        self.reach_time = None

    def close(self):
        self.simulator.end()

    def reset(self, *, seed=None, options=None):
        self.timer = 0

        self.target_handler.create_target()

        self.simulator.reset(target_pos=self.target_handler.position_target)

        measured_state = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(measured_state)

        self.distance_tool_target = None
        self.reach_time = None

        infos = {"dst": self.distance_tool_target, "time": self.reach_time}

        return (
            self.observation_handler.reset(
                measured_state,
                self.target_handler.position_target,
                None,
            ),
            infos,
        )

    def step(self, action):
        self.timer += 1

        reference_state = self.action_handler.compute_reference_state(action)
        reference_posture = reference_state[7 : 7 + self.n_joints]
        self.pd_controller.set_reference(reference_posture)

        torque_norm_sum = 0

        for _ in range(self.num_sim_steps):
            measured_state = self.simulator.getRobotState()
            torques = self.pd_controller.compute_torques(
                measured_state[7 : 7 + self.n_joints],
                measured_state[-self.n_joints :],
            )
            self.simulator.step(torques)

            torque_norm_sum += np.linalg.norm(torques)

        self.pinWrapper.update_reduced_model(measured_state)

        torque_norm_avg = torque_norm_sum / self.num_sim_steps

        terminated = self._is_terminated()
        truncated = self._is_truncated(measured_state, torques)
        observation = self.observation_handler.get_observation(
            measured_state,
            None,
        )
        reward = self._get_reward(torque_norm_avg, terminated)
        infos = {}

        return observation, reward, terminated, truncated, infos

    def _get_reward(self, avg_torque_norm, truncated):
        # Penalization of failure
        if truncated:
            reward_dead = -1
        else:
            reward_dead = 0

        # penalization of expanded energy
        reward_torque = -avg_torque_norm

        # distance to target
        self.distance_tool_target = np.linalg.norm(
            self.pinWrapper.get_end_effector_frame().translation
            - self.target_handler.position_target,
        )

        reward_distance = -self.distance_tool_target + 1

        # Success evaluation
        if self.distance_tool_target < self.distance_threshold:
            if self.reach_time is None:
                self.reach_time = self.timer

            reward_success = 1
        else:
            self.reach_time = None

            reward_success = 0

        return (
            self.weight_success * reward_success
            + self.weight_distance * reward_distance
            + self.weight_truncation * reward_dead
            + self.weight_energy * reward_torque
        )

    def _is_terminated(self):
        return self.timer > (self.max_step - 1)

    def _is_truncated(self, measured_state, torques):
        # Balance
        truncation_balance = (not (self.min_height == 0)) and (
            self.pinWrapper.get_com_position()[2] < self.min_height
        )

        # Limits
        limits = self.limit_checker.are_limits_broken(
            measured_state[7 : 7 + self.n_joints],
            measured_state[-self.n_joints :],
            torques,
        )

        if limits is not False:
            truncation_limits = True
        else:
            truncation_limits = False

        # Explicitly casting from numpy.bool_ to bool
        return bool(truncation_balance or truncation_limits)


class PDController:
    def __init__(self, controlled_joints, gains):
        self.reference_pos = np.zeros(len(controlled_joints))
        self.reference_vel = np.zeros(len(controlled_joints))

        self._set_gains(controlled_joints, gains)

    def _set_gains(self, controlled_joints, gains):
        self.Kp = np.zeros(len(controlled_joints))
        self.Kd = np.zeros(len(controlled_joints))

        for i, joint_name in enumerate(controlled_joints):
            self.Kp[i] = gains[joint_name]["kp"]
            self.Kd[i] = gains[joint_name]["kd"]

    def set_reference(self, reference_pos, reference_vel=None):
        if reference_vel is not None:
            self.reference_vel = reference_vel

        self.reference_pos = reference_pos

    def compute_torques(self, measured_pos, measured_vel):
        d_pos = measured_pos - self.reference_pos
        d_vel = measured_vel - self.reference_vel
        return -self.Kp * d_pos - self.Kd * d_vel
