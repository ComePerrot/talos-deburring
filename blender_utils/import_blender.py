import pickle
from pathlib import Path

import bpy
import mathutils

joint_mapping = {
    "leg_left_1_joint": 2,
    "leg_left_2_joint": 0,
    "leg_left_3_joint": 1,
    "leg_left_4_joint": 1,
    "leg_left_5_joint": 1,
    "leg_left_6_joint": 0,
    "leg_right_1_joint": 2,
    "leg_right_2_joint": 0,
    "leg_right_3_joint": 1,
    "leg_right_4_joint": 1,
    "leg_right_5_joint": 1,
    "leg_right_6_joint": 0,
    "torso_1_joint": 2,
    "torso_2_joint": 1,
    "arm_left_1_joint": 2,
    "arm_left_2_joint": 0,
    "arm_left_3_joint": 2,
    "arm_left_4_joint": 1,
    "arm_right_1_joint": 2,
    "arm_right_2_joint": 0,
    "arm_right_3_joint": 2,
    "arm_right_4_joint": 1,
}


class BlenderPoseHandler:
    def __init__(self, joint_mapping) -> None:
        self.joint_mapping = joint_mapping

    def load_trajectory(self, x_list):
        for i, x in enumerate(x_list):
            self.set_robot_pose(x, i)

    def set_robot_pose(self, x, frame_id):
        base_position = x[:3]
        base_quat = x[3:7]
        q = x[7:]

        self.set_base_placement(base_position, base_quat, frame_id)
        self.set_joint_state(q, frame_id)

    def set_base_placement(self, base_position, base_quat, frame_id):
        """Set the position and orientation of the base of the robot.

        Args:
            base_positiotn: 3D position of the base of the robot.
            base_quat: 4D quaternion orientation of the base of the robot.
        """
        blender_obj = bpy.data.objects["base_link"]

        # Set the base position
        blender_obj.location = base_position

        # Set the base orientation
        blender_quat = mathutils.Quaternion(
            (base_quat[-1], *base_quat[:-1])
        )  # Convert to Blender quaternion format
        blender_obj.rotation_mode = "QUATERNION"
        blender_obj.rotation_quaternion = blender_quat

        # Insert keyframes for position and orientation
        blender_obj.keyframe_insert("location", frame=frame_id)
        blender_obj.keyframe_insert("rotation_quaternion", frame=frame_id)

    def set_joint_state(self, q, frame_id):
        for (joint_name, axis_mapping), joint_value in zip(
            self.joint_mapping.items(), q
        ):
            blender_name = joint_name.replace("joint", "link")
            blender_obj = bpy.data.objects[blender_name]

            blender_obj.rotation_mode = "XYZ"
            blender_obj.rotation_euler[axis_mapping] = joint_value

            blender_obj.keyframe_insert("rotation_euler", frame=frame_id)


def load_joint_data(file_path):
    with Path.open(file_path, "rb") as file:
        return pickle.load(file)


# Example usage
file_path = Path(
    "/home/cperrot/talos-deburring/blender_utils/trajectories/trajectory_MPC.pkl"
)
x_list = load_joint_data(file_path)

pose_handler = BlenderPoseHandler(joint_mapping)

pose_handler.load_trajectory(x_list)
