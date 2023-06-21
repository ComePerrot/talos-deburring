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
        self._init_parameters(params_env)
    
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
            enableGUI=GUI,
            dt=self.timeStepSimulation,
        )

        action_dimension = self.rmodel.nq
        observation_dimension = len(self.simulator.getRobotState())
        self._init_env_variables(action_dimension, observation_dimension)

    def _init_parameters(self, params_env):
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

        self.torqueScale = np.array(self.rmodel.effortLimit)

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

    # def reset(self, seed: Optional[int] = None):
    #     super().reset(seed=seed)
    #     # Enforce that each GoalEnv uses a Goal-compatible observation space.
    #     if not isinstance(self.observation_space, gym.spaces.Dict):
    #         raise error.Error(
    #             "GoalEnv requires an observation space of type gym.spaces.Dict"
    #         )
    #     for key in ["observation", "achieved_goal", "desired_goal"]:
    #         if key not in self.observation_space.spaces:
    #             raise error.Error(
    #                 'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(
    #                     key
    #                 )
    #             )
            
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
        self.simulator.reset()
        self.targetPos = self._init_target(self.params_env) # Reset target position
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

        observation = self._getObservation(x_measured) # position and velocity of the joints and the final goal
        ob = observation["observation"]
        desired_goal = observation["desired_goal"]
        achieved_goal = observation["achieved_goal"]
        terminated = self._checkTermination(x_measured)
        truncated = self._checkTruncation(x_measured)
        reward, infos = self._getReward(torques, x_measured, terminated, truncated)
        return observation, reward, terminated, truncated, infos

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

    def _getRewardHER(self, torques, x_measured, terminated, truncated):
        """Compute step reward

        The reward is composed of:
            - A bonus when the environment is still alive (no constraint has been
              infriged)
            - A cost proportional to the norm of the torques
            - A cost proportional to the distance of the end-effector to the target

        Args:
            torques: torque vector
            x_measured: observation array obtained from the simulator
            terminated: termination bool
            truncated: truncation bool

        Returns: 
            Scalar reward
        """
        return 1 if np.linalg.norm(self.pinWrapper.get_end_effector_pos() - self.targetPos) < 0.2 else 0, {}
    
    def _getReward(self, torques, x_measured, terminated, truncated):
        """Compute step reward

        The reward is composed of:
            - A bonus when the environment is still alive (no constraint has been
              infriged)
            - A cost proportional to the norm of the torques
            - A cost proportional to the distance of the end-effector to the target

        Args:
            torques: torque vector
            x_measured: observation array obtained from the simulator
            terminated: termination bool
            truncated: truncation bool

        Returns:
            Scalar reward
        """
        if truncated:
            reward_alive = 0
        else:
            reward_alive = 1
        # command regularization
        reward_command = -np.linalg.norm(torques)
        # target distance
        reward_toolPosition = -np.linalg.norm(
            self.pinWrapper.get_end_effector_pos() - self.targetPos
        ) # + len_faster
        
        return (
            self.weight_target * reward_toolPosition
            + self.weight_command * reward_command
            + self.weight_truncation * reward_alive
        ), {
        }

    def _checkTermination(self, x_measured):
        """Check the termination conditions.

        Environment is terminated when the task has been successfully carried out.
        In our case it means that maxTime has been reached.

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            True if the environment has been terminated, False otherwise
        """
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
        truncation_balance = (not (self.minHeight == 0)) and (
            self.pinWrapper.CoM[2] < self.minHeight
        )

        # Limits
        truncation_limits_position = (
            x_measured[: self.rmodel.nq] > self.rmodel.upperPositionLimit
        ).any() or (x_measured[: self.rmodel.nq] < self.rmodel.lowerPositionLimit).any()
        truncation_limits_speed = (
            x_measured[-self.rmodel.nv :] > self.rmodel.velocityLimit
        ).any()
        truncation_limits = truncation_limits_position or truncation_limits_speed

        # Explicitely casting from numpy.bool_ to bool
        return bool(truncation_balance or truncation_limits)

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

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError
    
    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Compute the step termination. Allows to customize the termination states depending on the
        desired and the achieved goal. If you wish to determine termination states independent of the goal,
        you can include necessary values to derive it in 'info' and compute it accordingly. The envirtonment reaches
        a termination state when this state leads to an episode ending in an episodic task thus breaking .
        More information can be found in: https://farama.org/New-Step-API#theory

        Termination states are

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            bool: The termination state that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, terminated, truncated, info = env.step()
                assert terminated == env.compute_terminated(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError
    
    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Compute the step truncation. Allows to customize the truncated states depending on the
        desired and the achieved goal. If you wish to determine truncated states independent of the goal,
        you can include necessary values to derive it in 'info' and compute it accordingly. Truncated states
        are those that are out of the scope of the Markov Decision Process (MDP) such as time constraints in a
        continuing task. More information can be found in: https://farama.org/New-Step-API#theory

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            bool: The truncated state that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
        """
        raise NotImplementedError