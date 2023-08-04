import numpy as np
import pybullet as p  # PyBullet simulator
import pybullet_data
import pinocchio as pin


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
    ):
        self._setupBullet(enableGUI, enableGravity, dt)
        self._setupRobot(URDF, rmodelComplete, controlledJointsIDs, randomInit)
        self._setupInitializer(randomInit, rmodelComplete)
        self._createObjectVisuals()

    def _setupBullet(self, enableGUI, enableGravity, dt):
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
        self.initial_base_position = list(self.q0[:3])
        self.initial_base_orientation = list(self.q0[3:7])
        self.initial_joint_positions = list(self.q0[7:])

        rmodelComplete.armature = (
            rmodelComplete.rotorInertia * rmodelComplete.rotorGearRatio**2
        )

        # Load robot
        self.robotId = p.loadURDF(
            URDF,
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
            p.enableJointForceTorqueSensor(self.robotId, i, True)
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

        :param rmodelComplete Complete model of the robot
        :param ControlledJoints List of ControlledJoints
        """
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bullet_controlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bullet_controlledJoints],
        )

    def _setupInitializer(self, randomInit, rmodelComplete, noise_coeff=0.01):
        self.random_init = randomInit

        lower_limits_joint = rmodelComplete.lowerPositionLimit[7:]
        upper_limits_joint = rmodelComplete.upperPositionLimit[7:]
        amplitude_limits_joint = upper_limits_joint - lower_limits_joint

        lower_sampling_bound = (
            self.initial_joint_positions - noise_coeff * amplitude_limits_joint
        )
        upper_sampling_bound = (
            self.initial_joint_positions + noise_coeff * amplitude_limits_joint
        )
        self.lower_joint_bound = [
            max(lower_sampling_bound[i], lower_limits_joint[i])
            for i in range(len(self.initial_joint_positions))
        ]
        self.upper_joint_bound = [
            min(upper_sampling_bound[i], upper_limits_joint[i])
            for i in range(len(self.initial_joint_positions))
        ]
        self._init_joint_controlled(rmodelComplete)

    def _init_joint_controlled(self, rmodelComplete):
        """
        Initialize the joint controlled
        :param rmodelComplete Complete model of the robot
        """
        self.qC0 = np.empty(len(self.bullet_controlledJoints))
        self.dict_pos = {}
        for i in range(len(self.bullet_controlledJoints)):
            self.qC0[i] = p.getJointState(
                self.robotId,
                self.bullet_controlledJoints[i],
            )[0]
            self.dict_pos[
                rmodelComplete.names[
                    2
                    + self.bulletJointsIdInPinOrder.index(
                        self.bullet_controlledJoints[i],
                    )
                ]
            ] = i

    def _createObjectVisuals(self, target=True, tool=True, base=True):
        """Creates visual element for the simulation

        Args:
            target: Boolean to activate display of the target. Defaults to True.
            tool: Boolean to activate display of the tool. Defaults to True.
            base: Boolean to activate display of the base. Defaults to True.
        """
        RADIUS = 0.01
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
        redSphere = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[1, 0, 0, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS * 10,
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

        if base:
            self.base_visual = p.createMultiBody(
                baseMass=0.0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=redSphere,
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

    def getRobotPos(self):
        """Get current state of the robot from pyBullet"""
        # Get basis pose
        pos, quat = p.getBasePositionAndOrientation(self.robotId)
        # Get basis vel
        v, w = p.getBaseVelocity(self.robotId)
        # pos = np.array(pos) - self.localInertiaPos
        # Concatenate into a single x vector
        x = np.concatenate([pos, quat])
        # Magic transformation of the basis translation, as classical in Bullet.
        # x[:3] -= self.localInertiaPos

        return x  # noqa: RET504

    def getContactPoints(self):
        """Get contact points between the robot and the environment"""
        return [(i[4], i[6], i[9]) for i in p.getContactPoints()]

    def step(self, torques, oMtool=None, base_pose=None):
        """Do one step of simulation"""
        self._updateVisuals(oMtool, base_pose)
        self._applyTorques(torques)
        p.stepSimulation()
        self.baseRobot = np.array(
            [
                p.getBasePositionAndOrientation(self.robotId)[0][0],
                p.getBasePositionAndOrientation(self.robotId)[0][1],
                p.getBasePositionAndOrientation(self.robotId)[0][2],
            ],
        )

    def _updateVisuals(self, oMtool, base_pose):
        """Update visual elements of the simulation

        Args:
            oMtool: Placement of the tool expressed as a SE3 object
            base_pose: position of the base of the robot
        """
        if oMtool is not None:
            self._setVisualObjectPosition(
                self.tool_visual,
                oMtool.translation,
                pin.Quaternion(oMtool.rotation).coeffs(),
            )
        if base_pose is not None:
            self._setVisualObjectPosition(
                self.base_visual,
                base_pose,
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
            # self.physicsClient,
        )
        self.baseRobot = np.array(
            [
                p.getBasePositionAndOrientation(self.robotId)[0][0],
                p.getBasePositionAndOrientation(self.robotId)[0][1],
                p.getBasePositionAndOrientation(self.robotId)[0][2],
            ],
        )
        p.resetBaseVelocity(
            self.robotId,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            # self.physicsClient,
        )

        self._reset_robot_joints()
        self._setVisualObjectPosition(self.target_visual, target_pos)

    def _reset_robot_joints(self):
        for i in range(len(self.initial_joint_positions)):
            scale = 0.05
            if (
                self.bulletJointsIdInPinOrder[i] in self.bullet_controlledJoints
                and self.random_init
            ):
                init_pos = np.random.uniform(
                    low=self.lower_joint_bound[i],
                    high=self.upper_joint_bound[i],
                )
                self.qC0[
                    self.bullet_controlledJoints.index(self.bulletJointsIdInPinOrder[i])
                ] = init_pos
            else:
                init_pos = self.initial_joint_positions[i]
            p.resetJointState(
                self.robotId,
                self.bulletJointsIdInPinOrder[i],
                init_pos,
            )

    def end(self):
        """Ends connection with pybullet."""
        p.disconnect()
