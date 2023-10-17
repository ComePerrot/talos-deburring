import pinocchio as pin
import numpy as np
import yaml

from deburring_mpc import RobotDesigner

from gym_talos.utils import create_target

from simulator.bullet_Talos import TalosDeburringSimulator
from controllers.MPC import MPController
from controllers.RL_posture import RLPostureController


def main():
    target_handler = create_target()

    targetPos_1 = [0, 0, 0]

    filename = "settings_sobec.yaml"
    with open(filename, "r") as paramFile:
        params = yaml.safe_load(paramFile)

    modelPath = "/opt/openrobots/share/example-robot-data/robots/talos_data/"
    URDF = modelPath + "robots/talos_reduced.urdf"
    SRDF = modelPath + "srdf/talos.srdf"

    controlledJoints = params["robot"]["controlledJoints"]
    toolFramePos = params["robot"]["toolFramePos"]

    design_conf = dict(
        urdf_path=URDF,
        srdf_path=SRDF,
        left_foot_name="right_sole_link",
        right_foot_name="left_sole_link",
        robot_description="",
        controlled_joints_names=controlledJoints,
    )

    pinWrapper = RobotDesigner()
    pinWrapper.initialize(design_conf)

    gripper_SE3_tool = pin.SE3.Identity()
    gripper_SE3_tool.translation[0] = toolFramePos[0]
    gripper_SE3_tool.translation[1] = toolFramePos[1]
    gripper_SE3_tool.translation[2] = toolFramePos[2]

    pinWrapper.add_end_effector_frame(
        "deburring_tool", "gripper_left_fingertip_3_link", gripper_SE3_tool
    )

    oMtarget = pin.SE3.Identity()
    oMtarget.translation[0] = targetPos_1[0]
    oMtarget.translation[1] = targetPos_1[1]
    oMtarget.translation[2] = targetPos_1[2]

    oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    simulator = TalosDeburringSimulator(
        URDF=URDF,
        initialConfiguration=pinWrapper.get_q0_complete(),
        robotJointNames=pinWrapper.get_rmodel_complete().names,
        controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
        toolPlacement=pinWrapper.get_end_effector_frame(),
        targetPlacement=oMtarget,
    )

    kwargs_action = dict(
        True,
        pinWrapper.get_rmodel(),
        target_handler,
        0,
        0,
    )

    kwargs_observation = dict(
        None,
        pinWrapper.get_rmodel(),
        1,
        "full_range",
        None,
    )

    posture_controller = RLPostureController(
        "./test", kwargs_action, kwargs_observation
    )


if __name__ == "__main__":
    main()
