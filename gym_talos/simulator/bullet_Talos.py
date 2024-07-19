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

    def _setupBullet(self, enableGUI, enableGravity, dt):
        # Start the client for PyBullet
        if enableGUI:
            self.physicsClient = p.connect(p.SHARED_MEMORY)
            if (self.physicsClient < 0):
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
        self.lower_limits_joint = list(rmodelComplete.lowerPositionLimit[7:])
        self.upper_limits_joint = list(rmodelComplete.upperPositionLimit[7:])
        self.random_init = randomInit

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

        # Fetching the position of the center of mass
        # (which is different from the origin of the root link)
        self.localInertiaPos = p.getDynamicsInfo(self.robotId, -1)[3]

        # Expressing initial position wrt the CoM
        for i in range(3):
            self.initial_base_position[i] += self.localInertiaPos[i]

        self.names2bulletIndices = {
            p.getJointInfo(self.robotId, i)[1].decode(): i for i in range(p.getNumJoints(self.robotId))
        }
        self.bulletJointsIdInPinOrder = [
            self.names2bulletIndices[name] for name in rmodelComplete.names[2:]
        ]
        # Joints controlled with crocoddyl
        self.bullet_controlledJoints = [
            self.names2bulletIndices[rmodelComplete.names[i]]
            for i in controlledJointsIDs
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

        :param rmodelComplete Complete model of the robot
        :param ControlledJoints List of ControlledJoints
        """
        # Disable default position controler in torque controlled joints
        # Default controller will take care of other joints
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bullet_controlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bullet_controlledJoints],
        )

    def createTargetVisual(self, target):
        """Create visual representation of the target to track

        The visual will not appear unless the physics client is set to
        SHARED_MEMORY
        :param target Position of the target in the world
        """
        try:
            p.removeBody(self.target_MPC)
        except:
            pass
        RADIUS = 0.1
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
    
    def createCoMVisual(self):
        """Create visual representation of the CoM

        The visual will not appear unless the physics client is set to
        SHARED_MEMORY
        :param CoM Position of the CoM in the world
        """
        try:
            p.removeBody(self.CoM_MPC)
        except:
            pass
        RADIUS = 0.1
        LENGTH = 0.0001
        blueBox = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            rgbaColor=[1, 0, 0, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS,
            length=LENGTH,
            halfExtents=[0.0, 0.0, 0.0],
        )

        self.CoM_MPC = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=blueBox,
            basePosition=[self.CoM[0], self.CoM[1], self.CoM[2]],
            useMaximalCoordinates=True,
        )


    def getRobotState(self):
        """Get current state of the robot from pyBullet"""
        # Get articulated joint pos and vel
        xbullet = p.getJointStates(self.robotId, self.bullet_controlledJoints)
        q = [x[0] for x in xbullet]
        vq = [x[1] for x in xbullet]

        # Get basis pose
        pos, quat = p.getBasePositionAndOrientation(self.robotId)
        # Get basis vel
        v, w = p.getBaseVelocity(self.robotId)

        # Concatenate into a single x vector
        x = np.concatenate([q, vq])
        # Magic transformation of the basis translation, as classical in Bullet.
        # x[:3] -= self.localInertiaPos

        return x  # noqa: RET504
    
    def getRobotPos(self):
        """Get current state of the robot from pyBullet"""
        # Get basis pose
        pos, quat = p.getBasePositionAndOrientation(self.robotId)
        # Get basis vel
        v, w = p.getBaseVelocity(self.robotId)

        # Concatenate into a single x vector
        x = np.concatenate([pos, quat])
        # Magic transformation of the basis translation, as classical in Bullet.
        # x[:3] -= self.localInertiaPos

        return x  # noqa: RET504

    def step(self, torques):
        """Do one step of simulation"""
        self._applyTorques(torques)
        p.stepSimulation()
        self.CoM = np.array([p.getBasePositionAndOrientation(self.robotId)[0][0],
                             p.getBasePositionAndOrientation(self.robotId)[0][1], 
                             p.getBasePositionAndOrientation(self.robotId)[0][2]
                            ])

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
        self.createTargetVisual(target_position)
        p.resetBasePositionAndOrientation(
            self.robotId,
            self.initial_base_position,
            self.initial_base_orientation,
            # self.physicsClient,
        )
        p.resetBaseVelocity(
            self.robotId,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            # self.physicsClient,
        )
        # Reset joints
        for i in range(len(self.initial_joint_positions)):
            if self.bulletJointsIdInPinOrder[i] in self.bullet_controlledJoints and self.random_init:
                p.resetJointState(
                    self.robotId,
                    self.bulletJointsIdInPinOrder[i],
                    np.random.uniform(
                        low = 1/8 * (self.initial_joint_positions[i] + self.lower_limits_joint[i]),
                        high = 1/8 * (self.upper_limits_joint[i] + self.initial_joint_positions[i]),
                    ),
                )
            else:
                p.resetJointState(
                    self.robotId,
                    self.bulletJointsIdInPinOrder[i],
                    self.initial_joint_positions[i],
                )

    def end(self):
        """Ends connection with pybullet."""
        p.disconnect()
