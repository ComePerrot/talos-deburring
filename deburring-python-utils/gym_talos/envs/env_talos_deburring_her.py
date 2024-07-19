from typing import Dict, Tuple
import gymnasium as gym
import numpy as np
import collections
from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator

from ..utils.modelLoader import TalosDesigner
from ..utils.create_target import TargetGoal


class EnvTalosDeburringHer(gym.Env):
    def __init__(
        self,
        params_designer: Dict,
        params_env: Dict,
        GUI: bool = False,
    ) -> None:
        """Defines the EnvTalosDeburringHer class

        Defines an interface a robot designer to handle interactions with pinocchio,
        an interface to the simulator that will be used
        as well as usefull internal variables and can be used with the HER option if
        wanted

        Args:
            params_designer: kwargs for the robot designer
            params_env: kwargs for the environment
            GUI: set to true to activate display. Defaults to False.
        """
        # Environment parameters
        self._init_parameters(params_env=params_env, GUI=GUI)

        # Robot Designer
        self.pinWrapper = TalosDesigner(
            URDF=params_designer["URDF"],
            SRDF=params_designer["SRDF"],
            toolPosition=params_designer["toolPosition"],
            controlledJoints=params_designer["controlledJoints"],
            set_gravity=True,
            dt=self.timeStepSimulation * self.numSimulationSteps,
        )

        # Alias for the robot model
        self.rmodel = self.pinWrapper.rmodel

        # Simulator
        self.simulator = TalosDeburringSimulator(
            URDF=self.pinWrapper.URDF_path,
            rmodelComplete=self.pinWrapper.rmodelComplete,
            controlledJointsIDs=self.pinWrapper.controlledJointsID,
            randomInit=self.random_init_robot,
            enableGravity=True,
            enableGUI=GUI,
            dt=self.timeStepSimulation,
        )

        # Initialize the penalizations of the position of each joint to initial position
        self._init_penalization_position()

        # Initialize internal variables to shape the environment
        self._init_env_variables(
            action_dimension=self.rmodel.nq,
            observation_dimension=len(self.simulator.getRobotState()),
        )

    def _init_parameters(  # noqa: C901
        self,
        params_env: Dict,
        GUI: bool = False,
    ) -> None:
        """Load environment parameters from provided dictionnary

        Args:
            params_env: kwargs for the environment
            GUI: set to true to activate display. Defaults to False.
        """
        # Copy of the parameters (for callbacks)
        self.params_env = params_env

        # Simumlation timings
        self.timeStepSimulation = float(params_env["timeStepSimulation"])
        self.numSimulationSteps = params_env["numSimulationSteps"]

        self.normalizeObs = params_env["normalizeObs"]

        # Stop conditions
        self.maxTime = params_env["maxTime"]

        # Reward parameters
        self.weight_target = params_env["w_target_pos"]
        self.weight_command = params_env["w_control_reg"]
        self.weight_truncation = params_env["w_penalization_truncation"]
        try:
            self.weight_target_reached = params_env["w_target_reached"]
        except KeyError:
            self.weight_target_reached = 5
        try:
            self.weight_joints_to_init = params_env["w_joints_to_init"]
        except KeyError:
            self.weight_joints_to_init = None

        # Type of reward
        try:
            self.reward_type = params_env["rewardType"]
        except KeyError:
            self.reward_type = "dense"
        self._init_reward_type()

        # Time on target for success
        try:
            self.time_success = params_env["timeSuccess"]
        except KeyError:
            self.time_success = 100

        # Display
        self.GUI = GUI

        # Judgment to choose
        try:
            self.has_judgment = params_env["has_judgment"]
        except KeyError:
            self.has_judgment = False

        # Judgment, set to False at the begining for all cases
        self.is_able_to_judge = False

        # Random inits
        try:
            self.random_init_robot = params_env["randomInit"]
        except KeyError:
            self.random_init_robot = False

        # Limits of position and velocity and torques of Joints
        try:
            self.limitPosScale = params_env["limitPosScale"]
        except KeyError:
            self.limitPosScale = 10
        try:
            self.limitVelScale = params_env["limitVelScale"]
        except KeyError:
            self.limitVelScale = 30
        try:
            self.torqueScaleCoeff = params_env["torqueScaleCoeff"]
        except KeyError:
            self.torqueScaleCoeff = 1

        # Limits of position for the CoM of the robot
        try:
            self.lowerLimitPos = params_env["lowerLimitPos"]
        except KeyError:
            self.lowerLimitPos = [-0.5, -0.5, 0.9]
        try:
            self.upperLimitPos = params_env["upperLimitPos"]
        except KeyError:
            self.upperLimitPos = [0.5, 0.5, 1.5]

        # Threshold for success
        try:
            self.threshold_success = params_env["thresholdSuccess"]
        except KeyError:
            self.threshold_success = 0.05

        # Target builder
        self.target = TargetGoal(params_env=params_env)

    def _init_reward_type(self) -> None:
        """Initializes the reward type

        The reward can be dense, sparse or a mix of both.
        """
        if self.reward_type == "dense":
            self.compute_reward = self.compute_reward_dense
        elif self.reward_type == "sparse":
            self.compute_reward = self.compute_reward_sparse
        elif self.reward_type == "mix":
            self.compute_reward = self.compute_reward_mix

    def _init_penalization_position(self) -> None:
        """Initializes the penalization for the position

        The penalization for the position is computed using the distance to the
        initial position of the robot.
        """
        self.order_positions = self.simulator.dict_pos
        self.mat_dt_init = np.zeros(self.rmodel.nq)
        if self.weight_joints_to_init is not None:
            for key, value in self.weight_joints_to_init.items():
                self.mat_dt_init[self.order_positions[key]] = value
        self.mat_dt_init = np.diag(self.mat_dt_init)

    def _init_env_variables(self, action_dimension: int, observation_dimension: int):
        """Initialize internal variables of the environment

        Args:
            action_dimension: Dimension of the action space
            observation_dimension: Dimension of the observation space
        """
        self.maxStep = int(
            self.maxTime / (self.timeStepSimulation * self.numSimulationSteps),
        )

        if self.normalizeObs:
            self._init_obsNormalizer()
            self._init_goalNormalizer()
            self._init_targetNormalizer()
        self.torqueScale = self.torqueScaleCoeff * np.array(self.rmodel.effortLimit)
        if self.has_judgment:
            # Adding a dimension to the action space to activate a "button" that says
            # "i can't reach the target"
            # The perfect thing would be to add as last action a boolean
            # to copy a button like but didn't find the right solution with gym so here
            # it is a float and is acting like 1 if above 0 and 0 otherwise
            self.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(action_dimension + 1,),
                dtype=np.float32,
            )
        else:
            # Here no judgment is possible so no button added to the action space
            self.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(action_dimension,),
                dtype=np.float32,
            )

        # Having the required size of the observation space
        self.observation_space = gym.spaces.Dict()
        if self.normalizeObs:
            limit = 1
        else:
            limit = 10
        self.observation_space.spaces["observation"] = gym.spaces.Box(
            low=-limit,
            high=limit,
            shape=(observation_dimension,),
            dtype=np.float64,
        )
        self.observation_space.spaces["achieved_goal"] = gym.spaces.Box(
            low=-limit,
            high=limit,
            shape=(3,),
            dtype=np.float64,
        )
        self.observation_space.spaces["desired_goal"] = gym.spaces.Box(
            low=-limit,
            high=limit,
            shape=(3,),
            dtype=np.float64,
        )

    def close(self) -> None:
        """Properly shuts down the environment.

        Closes the simulator windows.
        """
        self.simulator.end()

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment

        Brings the robot back to its half-sitting position

        Args:
            seed: seed that is used to initialize the environment's PRNG.
                Defaults to None.
            options: Additional information can be specified to reset the environment.
                Defaults to None.
                target: Desired position of the target. If not specified, a random

        Returns:
            Observation of the initial state.
        """
        self.timer = 0
        self.time_on_target = 0
        if options is None:
            self.target.create_target()
        elif "target" in options.keys():
            self.target.set_target(options["target"])
        self.simulator.reset(self.target.position_target)  # Reset simulator
        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(
            x_measured,
            self.simulator.getContactPoints(),
            self.simulator.getRobotPos(),
        )
        infos = {
            "dst": np.linalg.norm(
                self.pinWrapper.get_end_effector_pos() - self.target.position_target,
            ),
            "tor": 0,
            "init": 0,
        }
        return self._getObservation(x_measured), infos

    def activate_judgment(self) -> None:
        """Activates the button

        Activates the button in the simulator
        """
        self.is_able_to_judge = True

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute a step of the environment

        One step of the environment is numSimulationSteps of the simulator with the same
        command.
        The model of the robot is updated using the observation taken from the
        environment.
        The termination and condition are checked and the reward is computed.
        Args:
            action: Normalized action vector

        Returns:
            Formatted observations
            Reward
            Boolean indicating this rollout is done
            Info dictionary, dst, tor, init, on_target, time_on_target, param_rew
        """
        self.timer += 1
        if self.has_judgment:
            torques = self._scaleAction(action[:-1])
            unreachable = False if not self.is_able_to_judge else action[-1] > 0
        else:
            torques = self._scaleAction(action)
            unreachable = False
        for _ in range(self.numSimulationSteps):
            self.simulator.step(
                torques,
                base_pose=self.pinWrapper.get_CoM(),
                oMtool=self.pinWrapper.get_end_effector_complete(),
            )
        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(
            x_measured,
            self.simulator.getContactPoints(),
            self.simulator.getRobotPos(),
        )
        ob = self._getObservation(x_measured)  # position velocity joint and goal
        truncated = self._checkTruncation(x_measured)
        reward, infos = self._reward(torques, ob, truncated, unreachable)
        terminated = self._checkTermination(unreachable)
        if terminated or truncated:
            infos["is_success"] = self._checkSuccess()
        return ob, reward, terminated, truncated, infos

    def _getObservation(self, x_measured: np.ndarray) -> collections.OrderedDict:
        """Formats observations

        Normalizes the observation obtained from the simulator if nomalizeObs = True

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            Formated observations
        """
        final_obs = gym.spaces.Dict()
        if self.normalizeObs:
            observation = self._obsNormalizer(x_measured)
            achieved_goal = self._goalNormalizer(self.pinWrapper.get_end_effector_pos())
            desired_goal = self._targetNormalizer(self.target.position_target)
        else:
            observation = x_measured
            achieved_goal = self.pinWrapper.get_end_effector_pos()
            desired_goal = self.target.position_target
        final_obs.spaces["observation"] = np.array(observation)
        final_obs.spaces["achieved_goal"] = np.array(achieved_goal)
        final_obs.spaces["desired_goal"] = np.array(desired_goal)
        return collections.OrderedDict(final_obs)

    def _reward(
        self,
        torques: np.ndarray,
        ob: np.ndarray,
        truncated: bool,
        unreachable: bool,
    ) -> Tuple[float, dict]:
        """Compute step reward

        The reward is composed of:
            - A bonus when the environment is still alive (no constraint has been
              infriged)
            - A cost proportional to the norm of the torques
            - A cost proportional to the distance of the end-effector to the target
            - A cost proportional to the distance of the robot to its initial position
            - A bonus if the target is reached

        Args:
            torques: torque vector
            ob: observation array obtained from the simulator
            terminated: termination bool
            truncated: truncation bool

        Returns:
            Scalar reward
        """
        # Length to initial position
        len_to_init = np.sum(
            (self.simulator.qC0 - ob["observation"][: self.rmodel.nq]).T
            * self.mat_dt_init
            * (self.simulator.qC0 - ob["observation"][: self.rmodel.nq]),
        )

        # Distance to target
        dst = np.linalg.norm(ob["achieved_goal"] - ob["desired_goal"])

        # Check if target is reached
        bool_check = dst < self.threshold_success

        # Adding infos, for comptuting and logs
        infos = {}
        infos["tor"] = np.linalg.norm(torques)
        infos["init"] = len_to_init
        infos["dst"] = dst
        infos["on_target"] = bool_check
        if infos["on_target"]:
            self.time_on_target += 1
        else:
            self.time_on_target = 0
        infos["time_on_target"] = self.time_on_target
        infos["param_rew"] = np.array(
            [np.linalg.norm(torques), len_to_init, not truncated],
        )
        ach_goal = np.empty((1, 3))
        des_goal = np.empty((1, 3))
        ach_goal[0, :] = ob["achieved_goal"]
        des_goal[0, :] = ob["desired_goal"]
        reward = float(
            self.compute_reward(
                achieved_goal=ach_goal,
                desired_goal=des_goal,
                info=np.array([infos]),
            ),
        )

        # Reward of 2 if the target is judged unreachable
        if unreachable:
            return 2, infos

        # Classic reward
        return reward, infos

    def _checkTermination(self, unreachable: bool) -> bool:
        """Check the termination conditions.

        Environment is terminated when the task has been successfully carried out.
        In our case it means that maxTime has been reached.

        Args:
            x_measured: observation array obtained from the simulator
            unreachable: bool indicating if the target is unreachable

        Returns:
            True if the environment has been terminated, Success sequence
            or target judged unreachable False otherwise
        """
        return (
            self.timer > (self.maxStep - 1)
            or self.time_on_target > self.time_success
            or unreachable
        )

    def _checkTruncation(self, x_measured: np.ndarray) -> bool:
        """Checks the truncation conditions.

        Environment is truncated when a constraint is infriged.
        There are two possible reasons for truncations:
         - Loss of balance of the robot:
            Rollout is stopped if position of CoM (or base currently) is under threshold
            No check is carried out if threshold is set to 0
         - Infrigement of the kinematic constraints of the robot
            Rollout is stopped if configuration exceeds model limits


        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            True if the environment has been truncated, False otherwise.
        """
        # Balance
        truncation_balance = (self.pinWrapper.get_CoM() < self.lowerLimitPos).any() or (
            self.pinWrapper.get_CoM() > self.upperLimitPos
        ).any()
        # Limits
        truncation_limits_position = (
            x_measured[: self.rmodel.nq]
            > self.limitPosScale * self.rmodel.upperPositionLimit
        ).any() or (
            x_measured[: self.rmodel.nq]
            < self.limitPosScale * self.rmodel.lowerPositionLimit
        ).any()
        truncation_limits_speed = (
            np.abs(x_measured[-self.rmodel.nv :])
            > self.limitVelScale * self.rmodel.velocityLimit
        ).any()
        truncation_limits = truncation_limits_position or truncation_limits_speed
        return truncation_limits or truncation_balance

    def _checkSuccess(self) -> bool:
        """Checks the success conditions.

        Environment is successful when the task has been successfully carried out.
        In our case it means that the end-effector is close enough to the target
        since a certain time.

        Returns:
            True if the episode has been successful, False otherwise.
        """
        return self.time_on_target > self.time_success

    def _scaleAction(self, action: np.ndarray):
        """Scales normalized actions to obtain robot torques

        Args:
            action: normalized action array

        Returns:
            torque array
        """

        return (
            self.torqueScale[6:] * action[7:]
            if self.simulator.has_free_flyer
            else self.torqueScale * action
        )

    def _init_obsNormalizer(self) -> None:
        """Initializes the observation normalizer using robot model limits"""
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

    def _init_goalNormalizer(self) -> None:
        """Initializes the goal normalizer using robot model limits"""
        self.lowerGoalLim = -3 * np.ones(3)
        self.upperGoalLim = 3 * np.ones(3)
        self.avgGoal = (self.upperGoalLim + self.lowerGoalLim) / 2
        self.diffGoal = self.upperGoalLim - self.lowerGoalLim

    def _init_targetNormalizer(self) -> None:
        """Initializes the target normalizer using robot model limits"""
        self.lowerGoalLim = -3 * np.ones(3)
        self.upperGoalLim = 3 * np.ones(3)
        self.avgGoal = (self.upperGoalLim + self.lowerGoalLim) / 2
        self.diffGoal = self.upperGoalLim - self.lowerGoalLim

    def _goalNormalizer(self, goal: np.ndarray) -> np.ndarray:
        """Normalizes the goal

        Args:
            goal: goal array

        Returns:
            normalized goal
        """
        return (goal - self.avgGoal) / self.diffGoal

    def _obsNormalizer(self, x_measured: np.ndarray) -> np.ndarray:
        """Normalizes the observation taken from the simulator

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            normalized observation
        """
        return (x_measured - self.avgObs) / self.diffObs

    def _targetNormalizer(self, target: np.ndarray) -> np.ndarray:
        """Normalizes the target

        Args:
            target: target array

        Returns:
            normalized target
        """
        return (target - self.avgGoal) / self.diffGoal

    def compute_reward_sparse(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        Compute the reward for the sparse reward case

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :return: the corresponding reward
        """
        dst = np.array([np.linalg.norm(achieved_goal - desired_goal, axis=-1)]).T
        coeff_matrix = np.array(
            [
                [
                    -self.weight_command,
                    # corresponds to the command penalization
                    0,
                    # corresponds to the lenght to init penalization
                    0,
                    # corresponds to the truncation penalization
                    self.weight_target_reached,
                    # corresponds to the target reached
                ],
            ],
        ).T
        info_matrix = np.empty((achieved_goal.shape[0], 3))

        for i, inf in enumerate(info):
            info_matrix[i] = inf["param_rew"]
        info_matrix = np.concatenate(
            (
                info_matrix,
                (dst < self.threshold_success).astype(int) - 0.002 * np.ones_like(dst),
            ),
            axis=1,
        )
        return info_matrix @ coeff_matrix

    def compute_reward_dense(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        Compute the reward for the dense reward case

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :return: the corresponding reward
        """

        dst = np.array([np.linalg.norm(achieved_goal - desired_goal, axis=-1)]).T
        coeff_matrix = np.array(
            [
                [
                    -self.weight_command,
                    # corresponds to the command penalization
                    -1,
                    # corresponds to the lenght to init penalization
                    self.weight_truncation,
                    # corresponds to the truncation penalization
                    -self.weight_target,
                    # corresponds to the distance to target
                ],
            ],
        ).T
        info_matrix = np.empty((achieved_goal.shape[0], 3))

        for i, inf in enumerate(info):
            info_matrix[i] = inf["param_rew"]
        info_matrix = np.concatenate(
            (info_matrix, dst),
            axis=1,
        )
        return info_matrix @ coeff_matrix

    def compute_reward_mix(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        Compute the reward for the mix reward case

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :return: the corresponding reward
        """

        dst = np.array([np.linalg.norm(achieved_goal - desired_goal, axis=-1)]).T
        bool_target = (dst < self.threshold_success).astype(int)
        coeff_matrix = np.array(
            [
                [
                    -self.weight_command,
                    # corresponds to the command penalization
                    -1,
                    # corresponds to the lenght to init penalization
                    self.weight_truncation,
                    # corresponds to the truncation penalization
                    -self.weight_target,
                    # corresponds to the distance to target
                    self.weight_target_reached,
                    # corresponds to the target reached
                ],
            ],
        ).T
        info_matrix = np.empty((achieved_goal.shape[0], 3))

        for i, inf in enumerate(info):
            info_matrix[i] = inf["param_rew"]
        info_matrix = np.concatenate(
            (info_matrix, dst),
            axis=1,
        )
        info_matrix = np.concatenate(
            (info_matrix, bool_target),
            axis=1,
        )
        return info_matrix @ coeff_matrix
