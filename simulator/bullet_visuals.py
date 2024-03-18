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
        """
        Set the position and orientation of a visual object.

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
        """
        Update the position and orientation of the tool visual object.

        Args:
            oMtool: The new SE3 pose of the tool.
        """
        if oMtool is not None:
            self.set_visual_object_position(
                self.tool_visual,
                oMtool.translation,
                pin.Quaternion(oMtool.rotation).coeffs(),
            )
