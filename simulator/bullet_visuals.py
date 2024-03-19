import numpy as np
import pinocchio as pin
import pybullet as p  # PyBullet simulator


class VisualHandler:
    """A class for handling visual elements in the PyBullet simulation.

    This class creates and manages visual objects such as the target and tool,
    and provides methods for updating their positions and orientations.

    Args:
        physics_client (int): The PyBullet physics client.
        target (bool, optional): Whether to create a visual target object. Defaults to True.
        tool (bool, optional): Whether to create a visual tool object. Defaults to True.
    """

    def __init__(self, physics_client, target=True, tool=True):
        """Initialize the VisualHandler class."""
        self.physics_client = physics_client
        self.RADIUS = 0.005
        self.LENGTH = 0.01
        self.create_visuals(target, tool)

    def create_visuals(self, target=True, tool=True):
        blue_sphere = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[0, 0, 1, 0.5],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=self.RADIUS,
            halfExtents=[0.0, 0.0, 0.0],
        )
        blue_capsule = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            rgbaColor=[0, 0, 1, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=self.RADIUS / 3,
            length=self.LENGTH,
            halfExtents=[0.0, 0.0, 0.0],
        )

        if target:
            self.target_visual = p.createMultiBody(
                baseMass=0.0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=blue_sphere,
                basePosition=[0.0, 0.0, 0.0],
                useMaximalCoordinates=True,
            )

        if tool:
            self.tool_visual = p.createMultiBody(
                baseMass=0.0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=blue_capsule,
                basePosition=[0.0, 0.0, 0.0],
                useMaximalCoordinates=True,
            )

    def set_visual_object_position(
        self,
        object_name,
        object_position,
        object_orientation=None,
    ):
        """Set the position and orientation of a visual object.

        Args:
            object_name: The name of the visual object to update.
            object_position: The new position of the visual object.
            object_orientation: The new orientation of the visual object. Defaults to None.
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

    def update_visuals(self, oMtool):
        """Update the position and orientation of the tool visual object.

        Args:
            oMtool: The new SE3 pose of the tool.
        """
        if oMtool is not None:
            self.set_visual_object_position(
                self.tool_visual,
                oMtool.translation,
                pin.Quaternion(oMtool.rotation).coeffs(),
            )

    def reset_visuals(self, target_pos):
        """Reset the position and orientation of the target

        Args:
            target_pos: The new position of the target.
        """
        self.set_visual_object_position(self.target_visual, target_pos)


class PostureVisualizer:
    """A class for handling reference posture visualization in the simulation.

    Args:
        URDF: Path to the URDF file of the robot.
        initial_base_position: Initial position of the base.
        initial_base_orientation: Initial orientation of the base.
        bullet_controlledJoints: List of joints controlled in torque (used to find size/order of the posture).
        initial_joint_positions: Initial joint configuration.
    """

    def __init__(
        self,
        URDF,
        initial_base_position,
        initial_base_orientation,
        bullet_controlledJoints,
        initial_joint_configuration,
    ):
        """Initialize the PostureVisualizer class."""
        # Load visual robot
        self.visual_robot = p.loadURDF(
            URDF,
            initial_base_position,
            initial_base_orientation,
            useFixedBase=True,
        )
        self.bullet_controlledJoints = bullet_controlledJoints
        # Set robot in initial pose
        for id_bullet, initial_pos in initial_joint_configuration.items():
            # p.enableJointForceTorqueSensor(self.robot_id, id_bullet, True)
            p.resetJointState(
                self.visual_robot,
                id_bullet,
                initial_pos,
            )

        # Change color and disable collisions
        color = [0, 0, 1, 0.2]  # second robot is gold
        # Base
        p.changeVisualShape(self.visual_robot, -1, rgbaColor=color)
        p.setCollisionFilterGroupMask(self.visual_robot, -1, 0, 0)

        # Joints
        for link_id in range(p.getNumJoints(self.visual_robot)):
            p.changeVisualShape(self.visual_robot, link_id, rgbaColor=color)
            p.setCollisionFilterGroupMask(self.visual_robot, link_id, 0, 0)

    def update_posture(self, posture):
        """Update the visual representation of the posture

        Args:
            posture: Reference joint posture of the robot (free-flyer should not be included)
        """
        for i, joint_id in enumerate(self.bullet_controlledJoints):
            p.resetJointState(
                self.visual_robot,
                joint_id,
                posture[i],
            )
