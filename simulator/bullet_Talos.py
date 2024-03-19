import numpy as np
import pybullet as p  # PyBullet simulator
import pybullet_data

from simulator.bullet_visuals import VisualHandler, PostureVisualizer
from simulator.filter import LowpassFilter
from simulator.pd_controller import PDController


class TalosDeburringSimulator:
    """Simulator class for Talos deburring task using PyBullet.

    Args:
        URDF: Path to the URDF file of the robot.
        rmodelComplete: Pinocchio model of the full robot.
        controlledJointsIDs: List of joint IDs to control in torque.
        enableGUI: Whether to enable PyBullet GUI. Defaults to False.
        enableGravity: Whether to enable gravity in the simulation. Defaults to True.
        dt: Time step of the simulation. Defaults to 1e-3.
        cutoff_frequency: Cutoff frequency for torque filtering. Defaults to 0.
    """

    def __init__(
        self,
        URDF,
        rmodelComplete,
        controlledJointsIDs,
        enableGUI=False,
        enableGravity=True,
        dt=1e-3,
        cutoff_frequency=0,
    ):
        """Initialize the simulator."""
        self._setup_client(enableGUI, enableGravity, dt)
        self._setup_robot(URDF, rmodelComplete, controlledJointsIDs)
        self._setup_PD_controller()
        self._setup_filter(cutoff_frequency, dt)
        self.visual_handler = VisualHandler(self.physics_client)
        self.posture_visualizer = PostureVisualizer(
            URDF,
            np.array(self.initial_base_position) - np.array(self.local_inertia_pos),
            self.initial_base_orientation,
            self.bullet_controlledJoints,
            self.initial_joint_positions,
        )

    def _setup_client(self, enableGUI, enableGravity, dt):
        """Set up PyBullet client and environment settings.

        Args:
            enableGUI: Whether to enable PyBullet GUI.
            enableGravity: Whether to enable gravity in the simulation.
            dt: Time step of the simulation.
        """
        # Start the client for PyBullet
        if enableGUI:
            self.physics_client = p.connect(p.SHARED_MEMORY)
            if self.physics_client < 0:
                self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        if enableGravity:
            p.setGravity(0, 0, -9.81)
        else:
            p.setGravity(0, 0, 0)

        p.setTimeStep(dt)

        # Load horizontal plane for PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

    def _setup_robot(self, URDF, rmodelComplete, controlledJointsIDs):
        """Set up the robot model and configuration.

        Args:
            URDF: Path to the URDF file of the robot.
            rmodelComplete: Pinocchio model of the full robot.
            controlledJointsIDs: List of joint IDs to control in torque.
        """
        self.q0 = rmodelComplete.referenceConfigurations["half_sitting"].copy()
        # Modify the height of the robot to avoid collision with the ground
        self.q0[2] += 0.01
        self.initial_base_position = list(self.q0[:3])
        self.initial_base_orientation = list(self.q0[3:7])
        self.initial_joint_positions = list(self.q0[7:])

        # Load robot
        self.robot_URDF = URDF
        self.robot_id = p.loadURDF(
            self.robot_URDF,
            self.initial_base_position,
            self.initial_base_orientation,
            useFixedBase=False,
        )

        # Fetching the position of the center of mass of the base
        # (which is different from the origin of the root link)
        self.local_inertia_pos = p.getDynamicsInfo(self.robot_id, -1)[3]
        # Expressing initial position wrt the CoM
        for i in range(3):
            self.initial_base_position[i] += self.local_inertia_pos[i]

        self.names2bulletIndices = {
            p.getJointInfo(self.robot_id, i)[1].decode(): i
            for i in range(p.getNumJoints(self.robot_id))
        }

        # Checks if the robot has a free-flyer to know how to shape state
        self.has_free_flyer = (
            rmodelComplete.getFrameId("root_joint") in controlledJointsIDs
        )
        # Needs to remove root_joint from joints controller by bullet
        # if robot has a free-flyer
        offset_controlled_joints = int(self.has_free_flyer) * 1

        self.bulletJointsIdInPinOrder = [
            self.names2bulletIndices[name] for name in rmodelComplete.names[2:]
        ]
        # Joints controlled with crocoddyl
        self.bullet_controlledJoints = [
            self.names2bulletIndices[rmodelComplete.names[i]]
            for i in controlledJointsIDs[offset_controlled_joints:]
        ]
        self._set_initial_config()
        self._change_friction(["leg_left_6_joint", "leg_right_6_joint"], 100, 30)
        self._set_controlled_joints()

    def _set_initial_config(self):
        """Initialize robot configuration in PyBullet."""
        for i in range(len(self.initial_joint_positions)):
            # p.enableJointForceTorqueSensor(self.robot_id, i, True)
            p.resetJointState(
                self.robot_id,
                self.bulletJointsIdInPinOrder[i],
                self.initial_joint_positions[i],
            )

    def _change_friction(self, names, lateral_friction=100, spinning_friction=30):
        """Change friction parameters for specified links.

        Args:
            names: List of link names to change friction for.
            lateral_friction: Lateral friction coefficient.
            spinning_friction: Spinning friction coefficient.
        """
        for n in names:
            idx = self.names2bulletIndices[n]
            p.changeDynamics(
                self.robot_id,
                idx,
                lateralFriction=lateral_friction,
                spinningFriction=spinning_friction,
            )

    def _set_controlled_joints(self):
        """Define torque controlled joints."""
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=self.bullet_controlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bullet_controlledJoints],
        )

    def _setup_filter(self, cutoff_frequency, dt):
        """Set up low-pass filter for torque control.

        Args:
            cutoff_frequency: Cutoff frequency for torque filtering.
            dt: Time step of the simulation.
        """
        if cutoff_frequency > 0:
            self.is_torque_filtered = True
            self.torque_filter = LowpassFilter(
                cutoff_frequency,
                1 / dt,
                len(self.bullet_controlledJoints),
            )
        else:
            self.is_torque_filtered = False

    def _setup_PD_controller(self):
        """Set up PD controller"""
        self.pd_controller = PDController()

    def getRobotState(self):
        """Get current state of the robot from PyBullet."""
        # Get articulated joint pos and vel
        xbullet = p.getJointStates(self.robot_id, self.bullet_controlledJoints)
        q = [x[0] for x in xbullet]
        vq = [x[1] for x in xbullet]

        if self.has_free_flyer:
            # Get base pose
            pos, quat = p.getBasePositionAndOrientation(self.robot_id)

            # Get base vel
            v, w = p.getBaseVelocity(self.robot_id)

            # Concatenate into a single x vector
            x = np.concatenate([pos, quat, q, v, w, vq])

            # Transformation between CoM of the base (base position in bullet)
            # and position of the base in Pinocchio
            x[:3] -= self.local_inertia_pos

        else:
            x = np.concatenate([q, vq])

        return x

    def step(self, torques, oMtool=None):
        """Do one step of simulation.

        Args:
            torques: Torques to be applied to the robot.
            oMtool: Placement of the tool expressed as a SE3 object.
        """
        self.visual_handler.update_visuals(oMtool)
        if self.is_torque_filtered:
            filtered_torques = self.torque_filter.filter(torques)
        else:
            filtered_torques = torques
        self._apply_torques(filtered_torques)
        p.stepSimulation()

    def _apply_torques(self, torques):
        """Apply computed torques to the robot.

        Args:
            torques: Torques to be applied to the robot.
        """
        p.setJointMotorControlArray(
            self.robot_id,
            self.bullet_controlledJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
        )

    def reset(self, target_pos, nb_pd_steps=0):
        """Reset robot to initial configuration.

        Args:
            target_pos: Position of the target.
            nb_pd_steps: Number of pd controlled steps to execute after reset. Defaults to 0.
        """
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.initial_base_position,
            self.initial_base_orientation,
            self.physics_client,
        )
        p.resetBaseVelocity(
            self.robot_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            self.physics_client,
        )

        self._set_initial_config()

        self.visual_handler.set_visual_object_position(
            self.visual_handler.target_visual,
            target_pos,
        )

        # Optionnal init phase with PD control
        for _ in range(nb_pd_steps):
            for id_bullet in self.bullet_controlledJoints:
                joint_name = p.getJointInfo(self.robot_id, id_bullet)[1].decode()
                joint_torque = self.pd_controller.compute_control(
                    joint_name,
                    p.getJointState(self.robot_id, id_bullet)[0],
                    p.getJointState(self.robot_id, id_bullet)[1],
                )
                p.setJointMotorControl(
                    self.robot_id,
                    id_bullet,
                    p.TORQUE_CONTROL,
                    joint_torque,
                )
            p.stepSimulation()

    def end(self):
        """End connection with PyBullet."""
        p.disconnect()
