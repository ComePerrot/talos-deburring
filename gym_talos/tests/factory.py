import example_robot_data
import pinocchio as pin
import pathlib
import yaml

from deburring_mpc import RobotDesigner
from gym_talos.utils.create_target import TargetGoal


class RobotModelFactory:
    def __init__(self):
        self._get_parameters()

        self.pinWrapper = RobotDesigner()
        self.pinWrapper.initialize(self.params_designer)

        gripper_SE3_tool = pin.SE3.Identity()
        gripper_SE3_tool.translation[0] = self.params_designer["toolPosition"][0]
        gripper_SE3_tool.translation[1] = self.params_designer["toolPosition"][1]
        gripper_SE3_tool.translation[2] = self.params_designer["toolPosition"][2]
        self.pinWrapper.add_end_effector_frame(
            "deburring_tool",
            "gripper_left_fingertip_3_link",
            gripper_SE3_tool,
        )

        self.rmodel = self.pinWrapper.get_rmodel()

        self.target_handler = TargetGoal(self.params_env)

    def _get_parameters(self):
        config_filename = pathlib.Path(__file__).with_name("config_test.yaml")
        with config_filename.open() as config_file:
            parameters = yaml.safe_load(config_file)
        self.params_designer = parameters["robot"]["designer"]

        self.params_designer["urdf_path"] = self._get_robot_urdf()
        self.params_designer["srdf_path"] = self._get_robot_srdf()

        self.params_env = parameters["environment"]

    def _get_robot_urdf(self):
        URDF = "/talos_data/robots/talos_reduced.urdf"
        modelPath = example_robot_data.getModelPath(URDF)
        return modelPath + URDF

    def _get_robot_srdf(self):
        SRDF = "/talos_data/srdf/talos.srdf"
        modelPath = example_robot_data.getModelPath(SRDF)
        return modelPath + SRDF

    def get_rmodel(self):
        return self.rmodel

    def get_target_handler(self):
        return self.target_handler
