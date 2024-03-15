import numpy as np
import pybullet as p  # PyBullet simulator
import pybullet_data
import pinocchio as pin

from .filter import LowpassFilter


class TalosDeburringSimulator:
    def __init__(
        self,
        URDF,
        rmodelComplete,
        controlledJointsIDs,
        randomInit=False,
        enableGUI=False,
        enableGravity=True,
        dt=1e-3,
        cutoff_frequency=0,
    ):
        self._setupClient(enableGUI, enableGravity, dt)
        self._setupRobot(URDF, rmodelComplete, controlledJointsIDs, randomInit)
        self._setupPDC()
        self._createObjectVisuals()

        if cutoff_frequency > 0:
            self.is_torque_filtered = True
            self.torque_filter = LowpassFilter(
                cutoff_frequency,
                1 / dt,
                len(self.bullet_controlledJoints),
            )
        else:
            self.is_torque_filtered = False

    def _setupClient(self, enableGUI, enableGravity, dt):
        # Start the client for PyBullet
        if enableGUI:
            self.physicsClient = p.connect(p.SHARED_MEMORY)
            if self.physicsClient < 0:
                self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # Set gravity (enabled by default)
        if enableGravity:
            p.setGravity(0, 0, -9.81)
        else:
            p.setGravity(0, 0, 0)

        # Set time step of the simulation
        p.setTimeStep(dt)

        # Load horizontal plane for PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

    def _setupRobot(self, URDF, rmodelComplete, controlledJointsIDs, randomInit):
        self.q0 = rmodelComplete.referenceConfigurations["half_sitting"]
        self.q0[2] += 0.01
        self.initial_base_position = list(self.q0[:3])
        self.initial_base_orientation = list(self.q0[3:7])
        self.initial_joint_positions = list(self.q0[7:])

        # Load robot
        self.robot_URDF = URDF
        self.robotId = p.loadURDF(
            self.robot_URDF,
            self.initial_base_position,
            self.initial_base_orientation,
            useFixedBase=False,
        )

        # Fetching the position of the center of mass of the base
        # (which is different from the origin of the root link)
        self.localInertiaPos = p.getDynamicsInfo(self.robotId, -1)[3]
        # Expressing initial position wrt the CoM
        for i in range(3):
            self.initial_base_position[i] += self.localInertiaPos[i]

        self.names2bulletIndices = {
            p.getJointInfo(self.robotId, i)[1].decode(): i
            for i in range(p.getNumJoints(self.robotId))
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
        """Initialize robot configuration in pyBullet

        :param q0 Intial robot configuration
        """
        for i in range(len(self.initial_joint_positions)):
            # p.enableJointForceTorqueSensor(self.robotId, i, True)
            p.resetJointState(
                self.robotId,
                self.bulletJointsIdInPinOrder[i],
                self.initial_joint_positions[i],
            )

    def _changeFriction(self, names, lateralFriction=100, spinningFriction=30):
        for n in names:
            idx = self.names2bulletIndices[n]
            p.changeDynamics(
                self.robotId,
                idx,
                lateralFriction=lateralFriction,
                spinningFriction=spinningFriction,
            )

    def _setControlledJoints(self):
        """Define joints controlled by pyBullet

        Disable default position controller in torque controlled joints.
        Default controller will take care of other joints.
        """
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bullet_controlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bullet_controlledJoints],
        )

    def _setupPDC(self):
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

    def _createObjectVisuals(self, target=True, tool=True):
        """Creates visual element for the simulation

        Args:
            target: Boolean to activate display of the target. Defaults to True.
            tool: Boolean to activate display of the tool. Defaults to True.
            base: Boolean to activate display of the base. Defaults to True.
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
        """Move an object to the given position

        Arguments:
            object_name -- Name of the object to move
            object_position -- Position of the object in the world
            object_orientation -- Quaternion representating the orientation of
                the object wrt the world
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
        """Get current state of the robot from pyBullet"""
        # Get articulated joint pos and vel
        xbullet = p.getJointStates(self.robotId, self.bullet_controlledJoints)
        q = [x[0] for x in xbullet]
        vq = [x[1] for x in xbullet]

        if self.has_free_flyer:
            # Get base pose
            pos, quat = p.getBasePositionAndOrientation(self.robotId)

            # Get base vel
            v, w = p.getBaseVelocity(self.robotId)

            # Concatenate into a single x vector
            x = np.concatenate([pos, quat, q, v, w, vq])

            # Transformation between CoM of the base (base position in bullet)
            # and position of the base in Pinocchio
            x[:3] -= self.localInertiaPos

        else:
            x = np.concatenate([q, vq])

        return x

    def step(self, torques, oMtool=None):
        """Do one step of simulation"""
        self._updateVisuals(oMtool)
        if self.is_torque_filtered:
            filtered_torques = self.torque_filter.filter(torques)
        else:
            filtered_torques = torques
        self._applyTorques(filtered_torques)
        p.stepSimulation()

    def _updateVisuals(self, oMtool):
        """Update visual elements of the simulation

        Args:
            oMtool: Placement of the tool expressed as a SE3 object
        """
        if oMtool is not None:
            self._setVisualObjectPosition(
                self.tool_visual,
                oMtool.translation,
                pin.Quaternion(oMtool.rotation).coeffs(),
            )

    def _applyTorques(self, torques):
        """Apply computed torques to the robot"""
        p.setJointMotorControlArray(
            self.robotId,
            self.bullet_controlledJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
        )

    def reset(self, target_pos, seed=None):
        """Reset robot to initial configuration"""
        # Reset base
        p.resetBasePositionAndOrientation(
            self.robotId,
            self.initial_base_position,
            self.initial_base_orientation,
            self.physicsClient,
        )
        p.resetBaseVelocity(
            self.robotId,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            self.physicsClient,
        )

        self._setInitialConfig()
        self._setVisualObjectPosition(self.target_visual, target_pos)

        # for _ in range(100):
        #     self.pd_controller()
        #     p.stepSimulation()

    def pd_controller(self):
        for id_pin, id_bullet in enumerate(self.bulletJointsIdInPinOrder):
            if id_bullet not in self.bullet_controlledJoints:
                continue

            joint_name = p.getJointInfo(self.robotId, id_bullet)[1].decode()

            d_pos = (
                p.getJointState(self.robotId, id_bullet)[0]
                - self.initial_joint_positions[id_pin]
            )
            d_vel = p.getJointState(self.robotId, id_bullet)[1]

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
                self.robotId,
                id_bullet,
                p.TORQUE_CONTROL,
                torque,
            )

    def end(self):
        """Ends connection with pybullet."""
        p.disconnect()
