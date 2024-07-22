import gymnasium as gym
import numpy as np
import pinocchio as pin

from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from deburring_mpc import RobotDesigner
from gym_talos.utils.action_wrapper import ActionWrapper
from gym_talos.utils.create_target import TargetGoal
from gym_talos.utils.observation_wrapper import observation_wrapper
from limit_checker_talos.limit_checker import LimitChecker
from robot_description.path_getter import srdf_path, urdf_path
from simulator.bullet_Talos import TalosDeburringSimulator


class EnvTalosMPC(gym.Env):
    def __init__(self, params_robot, params_env, GUI=False) -> None:
        params_designer = params_robot["designer"]
        self.param_ocp = params_robot["OCP"]

        self._init_parameters(params_env, self.param_ocp)

        # target
        self.target_handler = TargetGoal(params_env)
        self.target_handler.create_target()

        # Robot model handler
        self.pinWrapper = RobotDesigner()
        params_designer["end_effector_position"] = np.array(
            params_designer["end_effector_position"],
        )
        params_designer["urdf_path"] = urdf_path[params_designer["urdf_type"]]
        params_designer["srdf_path"] = srdf_path
        self.pinWrapper.initialize(params_designer)

        self.rmodel = self.pinWrapper.get_rmodel()

        self.limit_checker = LimitChecker(
            self.pinWrapper.get_rmodel(),
            extra_limits=params_env["extra_limits"],
            verbose=False,
        )

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
            rmodel_complete=self.pinWrapper.get_rmodel_complete(),
            controlled_joints_ids=self.pinWrapper.get_controlled_joints_ids(),
            enable_GUI=GUI,
            dt=self.timeStepSimulation,
        )

        observation_dimension = self.observation_handler.observation_size
        self._init_env_variables(self.action_dimension, observation_dimension)

        # action
        self.action_handler = ActionWrapper(
            self.rmodel,
            self.rl_controlled_joints,
            initial_state=self.q0,
            scaling_factor=params_env["actionScale"],
            scaling_mode=params_env["actionType"],
            clip_action=params_env["clipAction"],
        )

    def _init_ocp(self, param_ocp):
        self.oMtarget = pin.SE3.Identity()
        self.oMtarget.translation[0] = self.target_handler.position_target[0]
        self.oMtarget.translation[1] = self.target_handler.position_target[1]
        self.oMtarget.translation[2] = self.target_handler.position_target[2]

        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.mpc = MPController(
            self.pinWrapper,
            self.pinWrapper.get_x0(),
            self.target_handler.position_target,
            param_ocp,
            delay=param_ocp["delay"],
        )

        self.riccati = RiccatiController(
            state=self.mpc.crocoWrapper.state,
            torque=self.mpc.crocoWrapper.torque,
            xref=self.pinWrapper.get_x0(),
            riccati=self.mpc.crocoWrapper.gain,
        )

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
        self.sim_time = 0
        if options is None:
            options = {}

        if "target" in options.keys():
            self.target_handler.set_target(options["target"])
        else:
            self.target_handler.create_target()
        self.oMtarget.translation[0] = self.target_handler.position_target[0]
        self.oMtarget.translation[1] = self.target_handler.position_target[1]
        self.oMtarget.translation[2] = self.target_handler.position_target[2]

        self.simulator.reset(target_pos=self.target_handler.position_target)

        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(x_measured)

        self.mpc.change_target(self.pinWrapper.get_x0(), self.oMtarget.translation)

        self.distance_tool_target = None
        self.reach_time = None

        infos = {"dst": self.distance_tool_target, "time": self.reach_time}

        return (
            self.observation_handler.reset(
                x_measured,
                self.oMtarget.translation,
                self.mpc.crocoWrapper.solver.xs,
            ),
            infos,
        )

    def step(self, action):
        """Execute a step of the environment

        For each step of the environment the OCP is ran numOCPSteps times.
        The simulator is called numSimulationSteps between each iteration of the OCP.
        So each step of the environment represents numOCPSteps*numSimulationSteps calls to the simulator
        The model of the robot is updated using the observation taken from the
        environment.
        The termination and truncation conditions are checked and the reward is computed.

        Args:
            action: Normalized action vector (arm arm posture reference)

        Returns:
            Formatted observations
            Reward
            Boolean indicating this rollout is done
        """
        self.timer += 1
        x_measured = self.simulator.getRobotState()
        self.action_handler.update_initial_state(x_measured)

        posture_reference = self.action_handler.compute_reference_state(action)

        torque_sum = 0

        for _ in range(self.numOCPSteps * self.numSimulationSteps):
            x_measured = self.simulator.getRobotState()
            oMtool = self.pinWrapper.get_end_effector_frame()
            if self.sim_time % self.numSimulationSteps == 0:
                t0, x0, K0 = self.mpc.step(x_measured, posture_reference)
                self.riccati.update_references(t0, x0, K0)

            torques = self.riccati.step(x_measured)
            torque_sum += np.linalg.norm(torques)

            self.simulator.step(torques, oMtool)
            self.sim_time += 1

            if self._checkTruncation(x_measured, torques):
                break

        self.pinWrapper.update_reduced_model(x_measured)

        avg_torque_norm = torque_sum / (self.numOCPSteps * self.numSimulationSteps)

        observation = self.observation_handler.get_observation(
            x_measured,
            self.mpc.crocoWrapper.solver.xs,
        )
        terminated = self._checkTermination(x_measured)
        truncated = self._checkTruncation(x_measured, torques)
        reward = self._getReward(avg_torque_norm, x_measured, terminated, truncated)
        infos = {"dst": self.distance_tool_target, "time": self.reach_time}

        return observation, reward, terminated, truncated, infos

    def _getReward(self, avg_torque_norm, x_measured, terminated, truncated):
        """Compute step reward

        The reward is composed of:
        - A reward if the tool is close enough to the target
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

    def _checkTruncation(self, x_measured, torques):
        """Checks the truncation conditions.

        Environment is truncated when a constraint is infriged.
        There are two possible reasons for truncations:
        - Loss of balance of the robot:
            Rollout is stopped if position of CoM is under threshold
            No check is carried out if threshold is set to 0
        - Infrigement of the kinematic constraints of the robot
            Rollout is stopped safety limits are exceeded


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
        limits = self.limit_checker.are_limits_broken(
            x_measured[7 : self.rmodel.nq],
            x_measured[-self.rmodel.nv + 6 :],
            torques,
        )

        if limits is not False:
            truncation_limits = True
        else:
            truncation_limits = False

        # Explicitly casting from numpy.bool_ to bool
        return bool(truncation_balance or truncation_limits)
