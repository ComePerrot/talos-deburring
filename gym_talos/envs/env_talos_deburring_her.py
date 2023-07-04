import gymnasium as gym
import numpy as np

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator

from ..utils.modelLoader import TalosDesigner


class EnvTalosDeburringHer(gym.Env):
    def __init__(self, params_designer, params_env, GUI=False) -> None:
        """Defines the EnvTalosDeburring class

        Defines an interface a robot designer to handle interactions with pinocchio,
        an interface to the simulator that will be used
        as well as usefull internal variables.

        Args:
            params_designer: kwargs for the robot designer
            params_env: kwargs for the environment
            GUI: set to true to activate display. Defaults to False.
        """
        self._init_parameters(params_env, GUI)
    
        # Robot Designer
        self.pinWrapper = TalosDesigner(
            URDF=params_designer["URDF"],
            SRDF=params_designer["SRDF"],
            toolPosition=params_designer["toolPosition"],
            controlledJoints=params_designer["controlledJoints"],
        )

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

        action_dimension = self.rmodel.nq
        observation_dimension = len(self.simulator.getRobotState())
        self._init_env_variables(action_dimension, observation_dimension)

    def _init_parameters(self, params_env, GUI):
        """Load environment parameters from provided dictionnary

        Args:
            params_env: kwargs for the environment
        """
        # Simumlation timings
        self.timeStepSimulation = float(params_env["timeStepSimulation"])
        self.numSimulationSteps = params_env["numSimulationSteps"]

        self.normalizeObs = params_env["normalizeObs"]

        #   Stop conditions
        self.maxTime = params_env["maxTime"]
        self.minHeight = params_env["minHeight"]

        #   Target
        self.params_env = params_env
        self.targetType = params_env["targetType"]
        self.targetPos = self._init_target(params_env)

        #   Reward parameters
        self.weight_target = params_env["w_target_pos"]
        self.weight_command = params_env["w_control_reg"]
        self.weight_truncation = params_env["w_penalization_truncation"]
        
        self.GUI = GUI
        try:
            self.random_init_robot = params_env["randomInit"]
            self.limitPosScale = params_env["limitPosScale"]
            self.limitVelScale = params_env["limitVelScale"]
            self.torqueScaleCoeff = params_env["torqueScaleCoeff"]
            self.lowerLimitPos = params_env["lowerLimitPos"]
            self.upperLimitPos = params_env["upperLimitPos"]
        except KeyError:
            self.random_init_robot = False
            self.limitPosScale = 10
            self.limitVelScale = 30
            self.torqueScaleCoeff = 1
            self.lowerLimitPos = [-0.5, -0.5, 0.9]
            self.upperLimitPos = [0.5, 0.5, 1.5]

    def _init_target(self, param_env):
        """Initialize the target position

        Args:
            param_env: kwargs for the environment

        Returns:
            target_pos: target position
        """
        if self.targetType.lower() == "fixed":
            target_pos = param_env["targetPosition"]
        elif self.targetType.lower() == "reachable":
            phi   = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            u     = np.random.uniform(0, 1)
            target_pos = [param_env["shoulderPosition"][0] + 
                          u * np.sin(theta) * np.cos(phi), 
                          param_env["shoulderPosition"][1] +
                          u * np.sin(theta) * np.sin(phi), 
                          param_env["shoulderPosition"][2] +
                          u * np.cos(theta)]
        elif self.targetType.lower() == "box":
            size_low = param_env["targetSizeLow"]
            size_high = param_env["targetSizeHigh"]
            target_pos = [param_env["shoulderPosition"][0] +
                            np.random.uniform(size_low[0], size_high[0]),
                            param_env["shoulderPosition"][1] +
                            np.random.uniform(size_low[1], size_high[1]),
                            param_env["shoulderPosition"][2] +
                            np.random.uniform(size_low[2], size_high[2])]
        elif self.targetType.lower() == "sphere":
            phi   = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            radius = param_env["targetRadius"]
            u     = np.random.uniform(0, radius)
            target_pos = [param_env["shoulderPosition"][0] +
                            u * np.sin(theta) * np.cos(phi),
                            param_env["shoulderPosition"][1] +
                            u * np.sin(theta) * np.sin(phi),
                            param_env["shoulderPosition"][2] +
                            u * np.cos(theta)]
        else:
            raise ValueError("Unknown target type")
        return target_pos

    def _init_env_variables(self, action_dimension, observation_dimension):
        """Initialize internal variables of the environment

        Args:
            action_dimension: Dimension of the action space
            observation_dimension: Dimension of the observation space
        """
        self.timer = 0

        self.maxStep = int(
            self.maxTime / (self.timeStepSimulation * self.numSimulationSteps),
        )

        if self.normalizeObs:
            self._init_obsNormalizer()

        self.torqueScale = self.torqueScaleCoeff * np.array(self.rmodel.effortLimit)
        action_dim = action_dimension
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(action_dim,),
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
                    shape=(len(self.targetPos),),
                    dtype=np.float64,
                )
        self.observation_space.spaces["desired_goal"] = gym.spaces.Box(
                    low=-limit,
                    high=limit,
                    shape=(len(self.targetPos),),
                    dtype=np.float64,
                )

    def close(self):
        """Properly shuts down the environment.

        Closes the simulator windows.
        """
        self.simulator.end()
            
    def reset(self, *, seed=None, options=None):
        """Reset the environment

        Brings the robot back to its half-sitting position

        Args:
            seed: seed that is used to initialize the environment's PRNG.
                Defaults to None.
            options: Additional information can be specified to reset the environment.
                Defaults to None.

        Returns:
            Observation of the initial state.
        """
        self.timer = 0
        self.obj_reached = 0
        self.targetPos = self._init_target(self.params_env) # Reset target position
        self.simulator.reset(self.targetPos) # Reset simulator
        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(x_measured)
        return self._getObservation(x_measured), {}                                    

    def step(self, action):
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
        """
        self.timer += 1
        if self.timer == 1:
            self.initialPos = self.pinWrapper.get_end_effector_pos()
        torques = self._scaleAction(action)

        for _ in range(self.numSimulationSteps):
            self.simulator.step(torques)
        x_measured = self.simulator.getRobotState()

        self.pinWrapper.update_reduced_model(x_measured)
        if self.GUI:
            print(self.simulator.CoM)
        # Possibility to add torque to observation? 
        ob = self._getObservation(x_measured) # position and velocity of the joints and the final goal
        truncated = self._checkTruncation(x_measured)
        reward, infos = self._reward(torques, ob, truncated)
        terminated = self._checkTermination(infos)
        if infos['is_success']:
            self.obj_reached += 1
        return ob, reward, terminated, truncated, infos

    def _getObservation(self, x_measured):
        """Formats observations

        Normalizes the observation obtained from the simulator if nomalizeObs = True

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            Fromated observations
        """
        final_obs = gym.spaces.Dict()
        if self.normalizeObs:
            observation = self._obsNormalizer(x_measured)
        else:
            observation = x_measured
        final_obs.spaces["observation"] = np.array(observation)
        final_obs.spaces["achieved_goal"] = np.array(self.pinWrapper.get_end_effector_pos())
        final_obs.spaces["desired_goal"] = np.array(self.targetPos)
        return final_obs
    
    def _reward(self, torques, ob, truncated):
        """Compute step reward

        The reward is composed of:
            - A bonus when the environment is still alive (no constraint has been
              infriged)
            - A cost proportional to the norm of the torques
            - A cost proportional to the distance of the end-effector to the target

        Args:
            torques: torque vector
            ob: observation array obtained from the simulator
            terminated: termination bool
            truncated: truncation bool

        Returns:
            Scalar reward
        """
        pos_reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'], {}, p=1)
        bool_check = np.abs(pos_reward) < 0.1
        infos = {}
        infos['is_success'] = bool_check
        reward = 4 * bool_check.astype(float)
        reward += self.weight_target * pos_reward
        reward += self.weight_truncation if not truncated else 0
        reward += self.weight_command * -np.linalg.norm(torques)
        return reward, infos
    
    def _checkTermination(self, infos):
        """Check the termination conditions.

        Environment is terminated when the task has been successfully carried out.
        In our case it means that maxTime has been reached.

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            True if the environment has been terminated, False otherwise
        """
        infos['time'] = self.timer
        return self.timer > (self.maxStep - 1) 

    def _checkTruncation(self, x_measured):
        """Checks the truncation conditions.

        Environment is truncated when a constraint is infriged.
        There are two possible reasons for truncations:
         - Loss of balance of the robot:
            Rollout is stopped if position of CoM is under threshold
            No check is carried out if threshold is set to 0
         - Infrigement of the kinematic constraints of the robot
            Rollout is stopped if configuration exceeds model limits


        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            True if the environment has been truncated, False otherwise.
        """
        # Balance
        truncation_balance =  (self.simulator.CoM < self.lowerLimitPos).any() or (self.simulator.CoM > self.upperLimitPos
        ).any()
        # Limits
        truncation_limits_position = (
            x_measured[: self.rmodel.nq] > self.limitPosScale * self.rmodel.upperPositionLimit
        ).any() or (x_measured[: self.rmodel.nq] < self.limitPosScale * self.rmodel.lowerPositionLimit).any()
        truncation_limits_speed = (
            np.abs(x_measured[-self.rmodel.nv :]) >  self.limitVelScale * self.rmodel.velocityLimit
        ).any()
        truncation_limits = truncation_limits_position or truncation_limits_speed

        # if truncation_limits_position:
        #     print("Limit due to position: ", x_measured[: self.rmodel.nq], 1 * self.rmodel.lowerPositionLimit, 1 * self.rmodel.upperPositionLimit)
        # if truncation_limits_speed:
        #     print("Limit due to speed: ", x_measured[-self.rmodel.nv :], np.abs(x_measured[-self.rmodel.nv:]),  1 * self.rmodel.velocityLimit)

        # Explicitely casting from numpy.bool_ to bool
        return truncation_limits or truncation_balance
    
    def _scaleAction(self, action):
        """Scales normalized actions to obtain robot torques

        Args:
            action: normalized action array

        Returns:
            torque array
        """
        return self.torqueScale * action

    def _init_obsNormalizer(self):
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

    def _obsNormalizer(self, x_measured):
        """Normalizes the observation taken from the simulator

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            normalized observation
        """
        return (x_measured - self.avgObs) / self.diffObs
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return - np.sum(np.power(np.abs(achieved_goal - desired_goal), p), axis=-1)