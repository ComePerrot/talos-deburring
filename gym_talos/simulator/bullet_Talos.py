import numpy as np
import pybullet as p  # PyBullet simulator
import pybullet_data


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

    def createTargetVisual(self, target):
        """Create visual representation of the target to track

        The visual will not appear unless the physics client is set to
        SHARED_MEMORY
        :param target Position of the target in the world
        """
        try:
            p.removeBody(self.target_MPC)
        except:  # noqa: E722
            pass
        RADIUS = 0.02
        LENGTH = 0.0001
        blueBox = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            rgbaColor=[0, 0, 1, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS,
            length=LENGTH,
            halfExtents=[0.0, 0.0, 0.0],
        )

        self.target_MPC = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=blueBox,
            basePosition=[target[0], target[1], target[2]],
            useMaximalCoordinates=True,
        )

    def createBaseRobotVisual(self, baseRobot):
        """Create visual representation of the CoM

        The visual will not appear unless the physics client is set to
        SHARED_MEMORY
        :param CoM Position of the CoM in the world
        :param baseRobot Position of the base of the robot in the world
        """
        try:
            p.removeBody(self.baseRobot_MPC)
        except:  # noqa: E722
            pass
        RADIUS = 0.03
        LENGTH = 0.0001
        blueBox = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            rgbaColor=[1, 0, 0, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS,
            length=LENGTH,
            halfExtents=[0.0, 0.0, 0.0],
        )

        self.baseRobot_MPC = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=blueBox,
            basePosition=[baseRobot[0], baseRobot[1], baseRobot[2]],
            useMaximalCoordinates=True,
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

    def step(self, torques):
        """Do one step of simulation"""
        self._applyTorques(torques)
        p.stepSimulation()
        self.baseRobot = np.array(
            [
                p.getBasePositionAndOrientation(self.robotId)[0][0],
                p.getBasePositionAndOrientation(self.robotId)[0][1],
                p.getBasePositionAndOrientation(self.robotId)[0][2],
            ],
        )

    def _applyTorques(self, torques):
        """Apply computed torques to the robot"""
        p.setJointMotorControlArray(
            self.robotId,
            self.bullet_controlledJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
        )

    def reset(self, target_position, seed=None):
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
        self.createTargetVisual(target_position)

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
