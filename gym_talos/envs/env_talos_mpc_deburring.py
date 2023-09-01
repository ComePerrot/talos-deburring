import gymnasium as gym
import numpy as np
import pinocchio as pin
from deburring_mpc import RobotDesigner, OCP

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator
from gym_talos.utils.create_target import TargetGoal
from gym_talos.utils.observation_wrapper import observation_wrapper


class EnvTalosMPC(gym.Env):
    def __init__(self, params_robot, params_env, GUI=False) -> None:
        params_designer = params_robot["designer"]
        self.param_ocp = params_robot["OCP"]
        self.param_ocp["state_weights"] = np.array(self.param_ocp["state_weights"])
        self.param_ocp["control_weights"] = np.array(self.param_ocp["control_weights"])

        self._init_parameters(params_env, self.param_ocp)

        # target
        self.target_handler = TargetGoal(params_env)
        self.target_handler.create_target()

        # Robot wrapper
        self.pinWrapper = RobotDesigner()
        self.pinWrapper.initialize(params_designer)

        gripper_SE3_tool = pin.SE3.Identity()
        gripper_SE3_tool.translation[0] = params_designer["toolPosition"][0]
        gripper_SE3_tool.translation[1] = params_designer["toolPosition"][1]
        gripper_SE3_tool.translation[2] = params_designer["toolPosition"][2]
        self.pinWrapper.add_end_effector_frame(
            "deburring_tool",
            "gripper_left_fingertip_3_link",
            gripper_SE3_tool,
        )

        self.rmodel = self.pinWrapper.get_rmodel()

        self.rl_controlled_IDs = np.array(
            [
                self.rmodel.names.tolist().index(joint_name) - 2 + 7
                for joint_name in self.rl_controlled_joints
            ],
        )
        self.action_dimension = len(self.rl_controlled_IDs)

        # OCP
        self._init_ocp(self.param_ocp)
        self.horizon_length = self.param_ocp["horizon_length"]

        # observation
        self.observation_handler = observation_wrapper(
            self.normalizeObs,
            self.rmodel,
            self.target_handler,
            self.historyObs,
            self.predictionSize,
        )

        # Simulator
        self.simulator = TalosDeburringSimulator(
            URDF=params_designer["urdf_path"],
            rmodelComplete=self.pinWrapper.get_rmodel_complete(),
            controlledJointsIDs=self.pinWrapper.get_controlled_joints_ids(),
            enableGUI=GUI,
            dt=self.timeStepSimulation,
        )

        observation_dimension = self.observation_handler.observation_size
        self._init_env_variables(self.action_dimension, observation_dimension)

        self.target_handler = TargetGoal(params_env)

    def _init_ocp(self, param_ocp):
        self.oMtarget = pin.SE3.Identity()
        self.oMtarget.translation[0] = self.target_handler.position_target[0]
        self.oMtarget.translation[1] = self.target_handler.position_target[1]
        self.oMtarget.translation[2] = self.target_handler.position_target[2]

        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.crocoWrapper = OCP(param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(self.pinWrapper.get_x0(), self.oMtarget)

        self.ddp = self.crocoWrapper.solver

        # self.X_warm = self.ddp.xs
        # self.U_warm = self.ddp.us

        # Problem can be modified here to fit the needs of the RL

    def _init_parameters(self, params_env, param_ocp):
        """Load environment parameters from provided dictionnary

        Args:
            params_env: kwargs for the environment
        """
        self.rl_controlled_joints = params_env["controlled_joints_names"]
        # Simulation timings
        self.timeStepSimulation = float(params_env["timeStepSimulation"])
        self.timeStepOCP = float(param_ocp["time_step"])

        self.numOCPSteps = params_env["numOCPSteps"]
        self.historyObs = params_env["historyObs"]
        self.predictionSize = params_env["predictionSize"]
        self.normalizeObs = params_env["normalizeObs"]

        #   Stop conditions
        self.maxTime = params_env["maxTime"]
        self.minHeight = params_env["minHeight"]

        #   Reward parameters
        self.distanceThreshold = params_env["distanceThreshold"]
        self.weight_success = params_env["w_success"]
        self.weight_distance = params_env["w_distance"]
        self.weight_truncation = params_env["w_penalization_truncation"]
        self.weight_energy = params_env["w_penalization_torque"]

    def _init_env_variables(self, action_dimension, observation_dimension):
        """Initialize internal variables of the environment

        Args:
            action_dimension: Dimension of the action space
            observation_dimension: Dimension of the observation space
        """
        self.timer = 0

        self.numSimulationSteps = int(self.timeStepOCP / self.timeStepSimulation)
        self.maxStep = int(
            self.maxTime
            / (self.numOCPSteps * self.timeStepSimulation * self.numSimulationSteps),
        )

        self._init_actScaler()

        self.q0 = self.pinWrapper.get_x0().copy()

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(action_dimension,),
            dtype=np.float32,
        )

        if self.normalizeObs:
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

        self.target_handler.create_target()
        self.oMtarget.translation[0] = self.target_handler.position_target[0]
        self.oMtarget.translation[1] = self.target_handler.position_target[1]
        self.oMtarget.translation[2] = self.target_handler.position_target[2]

        self.simulator.reset(target_pos=self.oMtarget.translation)

        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(x_measured)

        self._init_ocp(self.param_ocp)

        # self.crocoWrapper.reset(x_measured, self.oMtarget)
        # self.crocoWrapper.set_warm_start(self.X_warm, self.U_warm)

        self.distance_tool_target = None
        self.reach_time = None

        infos = {"dst": self.distance_tool_target, "time": self.reach_time}

        return (
            self.observation_handler.reset(
                x_measured,
                self.oMtarget.translation,
                self.crocoWrapper.solver.xs,
            ),
            infos,
        )

    def step(self, action):
        """Execute a step of the environment

        One step of the environment is numSimulationSteps of the simulator with the same
        command.
        The model of the robot is updated using the observation taken from the
        environment.
        The termination and condition are checked and the reward is computed.

        Args:
            action: Normalized action vector (arm arm posture reference)

        Returns:
            Formatted observations
            Reward
            Boolean indicating this rollout is done
        """
        self.timer += 1
        arm_reference = self._actScaler(action)

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

    def _getReward(self, avg_torque_norm, x_measured, terminated, truncated):
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
        if self.distance_tool_target < self.distanceThreshold:
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
            self.pinWrapper.get_com_position()[2] < self.minHeight
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

    def _init_actScaler(self):
        """Initializes the action scaler using robot model limits"""
        self.lowerActLim = np.array(
            [
                self.rmodel.lowerPositionLimit[joint_ID]
                for joint_ID in self.rl_controlled_IDs
            ],
        )
        self.upperActLim = np.array(
            [
                self.rmodel.upperPositionLimit[joint_ID]
                for joint_ID in self.rl_controlled_IDs
            ],
        )

        self.avgAct = (self.upperActLim + self.lowerActLim) / 2
        self.diffAct = self.upperActLim - self.lowerActLim

    def _actScaler(self, action):
        """Scale the action given by the agent

        Args:
            action: normalized action given by the agent

        Returns:
            unnormalized reference
        """
        return action * self.diffAct + self.avgAct
