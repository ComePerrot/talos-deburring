import numpy as np
import pybullet as p  # PyBullet simulator
import pybullet_data
import pinocchio as pin

from .filter import LowpassFilter


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
        self._create_visuals()

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
        self.q0 = rmodelComplete.referenceConfigurations["half_sitting"]
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
        self.localInertiaPos = p.getDynamicsInfo(self.robot_id, -1)[3]
        # Expressing initial position wrt the CoM
        for i in range(3):
            self.initial_base_position[i] += self.localInertiaPos[i]

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
        self._setInitialConfig()
        self._changeFriction(["leg_left_6_joint", "leg_right_6_joint"], 100, 30)
        self._setControlledJoints()

    def _setInitialConfig(self):
        """Initialize robot configuration in PyBullet."""
        for i in range(len(self.initial_joint_positions)):
            # p.enableJointForceTorqueSensor(self.robot_id, i, True)
            p.resetJointState(
                self.robot_id,
                self.bulletJointsIdInPinOrder[i],
                self.initial_joint_positions[i],
            )

    def _changeFriction(self, names, lateralFriction=100, spinningFriction=30):
        """Change friction parameters for specified links.

        Args:
            names: List of link names to change friction for.
            lateralFriction: Lateral friction coefficient.
            spinningFriction: Spinning friction coefficient.
        """
        for n in names:
            idx = self.names2bulletIndices[n]
            p.changeDynamics(
                self.robot_id,
                idx,
                lateralFriction=lateralFriction,
                spinningFriction=spinningFriction,
            )

    def _setControlledJoints(self):
        """Define torque controlled joints."""
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=self.bullet_controlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bullet_controlledJoints],
        )

    def _setup_PD_controller(self):
        """Set up PD controller gains and feed-forward terms."""
        self.p_arm_gain = 100.0
        self.d_arm_gain = 8.0
        self.p_torso_gain = 500.0
        self.d_torso_gain = 20.0
        self.p_leg_gain = 800.0
        self.d_leg_gain = 35.0
        self.feed_forward = {
            "leg_left_1_joint": 0,
            "leg_left_2_joint": 1,
            "leg_left_3_joint": 2,
            "leg_left_4_joint": -5e01,
            "leg_left_5_joint": 3,
            "leg_left_6_joint": -5,
            "leg_right_1_joint": 0,
            "leg_right_2_joint": 1,
            "leg_right_3_joint": 2,
            "leg_right_4_joint": -5e01,
            "leg_right_5_joint": 3,
            "leg_right_6_joint": 5,
            "torso_1_joint": 0,
            "torso_2_joint": 6e-1,
            "arm_left_1_joint": 6e-02,
            "arm_left_2_joint": 5e-01,
            "arm_left_3_joint": 2.2,
            "arm_left_4_joint": -9.3,
            "arm_left_5_joint": 1.1e-01,
            "arm_left_6_joint": 3.2e-01,
            "arm_left_7_joint": -1.5,
            "arm_right_1_joint": -6e-02,
            "arm_right_2_joint": -5e-01,
            "arm_right_3_joint": -2.2,
            "arm_right_4_joint": -9.3,
            "arm_right_5_joint": 1.1e-01,
            "arm_right_6_joint": 3.2e-01,
            "arm_right_7_joint": -1.5,
        }

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

    def _create_visuals(self, target=True, tool=True):
        """Create visual elements for the simulation.

        Args:
            target: Whether to display the target.
            tool: Whether to display the tool.
        """
        RADIUS = 0.005
        LENGTH = 0.01
        blueSphere = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[0, 0, 1, 0.5],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS,
            halfExtents=[0.0, 0.0, 0.0],
        )
        blueCapsule = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            rgbaColor=[0, 0, 1, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS / 3,
            length=LENGTH,
            halfExtents=[0.0, 0.0, 0.0],
        )

        if target:
            self.target_visual = p.createMultiBody(
                baseMass=0.0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=blueSphere,
                basePosition=[0.0, 0.0, 0.0],
                useMaximalCoordinates=True,
            )

        if tool:
            self.tool_visual = p.createMultiBody(
                baseMass=0.0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=blueCapsule,
                basePosition=[0.0, 0.0, 0.0],
                useMaximalCoordinates=True,
            )

    def _setVisualObjectPosition(
        self,
        object_name,
        object_position,
        object_orientation=None,
    ):
        """Move an object to the given position.

        Args:
            object_name: Name of the object to move.
            object_position: Position of the object in the world.
            object_orientation: Orientation of the object wrt the world.
        """
        if object_orientation is not None:
            p.resetBasePositionAndOrientation(
                object_name,
                object_position,
                object_orientation,
            )
        else:
            p.resetBasePositionAndOrientation(
                object_name,
                object_position,
                np.array([0, 0, 0, 1]),
            )

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
            x[:3] -= self.localInertiaPos

        else:
            x = np.concatenate([q, vq])

        return x

    def step(self, torques, oMtool=None):
        """Do one step of simulation.

        Args:
            torques: Torques to be applied to the robot.
            oMtool: Placement of the tool expressed as a SE3 object.
        """
        self._updateVisuals(oMtool)
        if self.is_torque_filtered:
            filtered_torques = self.torque_filter.filter(torques)
        else:
            filtered_torques = torques
        self._applyTorques(filtered_torques)
        p.stepSimulation()

    def _updateVisuals(self, oMtool):
        """Update visual elements of the simulation.

        Args:
            oMtool: Placement of the tool expressed as a SE3 object.
        """
        if oMtool is not None:
            self._setVisualObjectPosition(
                self.tool_visual,
                oMtool.translation,
                pin.Quaternion(oMtool.rotation).coeffs(),
            )

    def _applyTorques(self, torques):
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

    def reset(self, target_pos):
        """Reset robot to initial configuration.

        Args:
            target_pos: Position of the target.
        """
        # Reset base
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

        self._setInitialConfig()
        self._setVisualObjectPosition(self.target_visual, target_pos)

        # for _ in range(100):
        #     self.pd_controller()
        #     p.stepSimulation()

    def pd_controller(self):
        """Run PD controller for the robot."""
        for id_pin, id_bullet in enumerate(self.bulletJointsIdInPinOrder):
            if id_bullet not in self.bullet_controlledJoints:
                continue

            joint_name = p.getJointInfo(self.robot_id, id_bullet)[1].decode()

            d_pos = (
                p.getJointState(self.robot_id, id_bullet)[0]
                - self.initial_joint_positions[id_pin]
            )
            d_vel = p.getJointState(self.robot_id, id_bullet)[1]

            feed_forward = self.feed_forward[joint_name]

            if "torso" in joint_name:
                torque = (
                    feed_forward - self.p_torso_gain * d_pos - self.d_torso_gain * d_vel
                )

            elif "arm" in joint_name:
                torque = (
                    feed_forward - self.p_arm_gain * d_pos - self.d_arm_gain * d_vel
                )

            elif "leg" in joint_name:
                torque = (
                    feed_forward - self.p_leg_gain * d_pos - self.d_leg_gain * d_vel
                )

            else:
                torque = 0

            p.setJointMotorControl(
                self.robot_id,
                id_bullet,
                p.TORQUE_CONTROL,
                torque,
            )

    def end(self):
        """End connection with PyBullet."""
        p.disconnect()
