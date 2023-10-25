import pinocchio as pin
import numpy as np
import yaml

from deburring_mpc import RobotDesigner

from gym_talos.utils.create_target import TargetGoal

from simulator.bullet_Talos import TalosDeburringSimulator
from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from controllers.RL_posture import RLPostureController


def main():
    # PARAMETERS
    filename = "settings_sobec.yaml"
    with open(filename, "r") as paramFile:
        params = yaml.safe_load(paramFile)

    target_handler = TargetGoal(params["target"])
    target_handler.create_target()
    target = target_handler.position_target

    oMtarget = pin.SE3.Identity()
    oMtarget.translation[0] = target[0]
    oMtarget.translation[1] = target[1]
    oMtarget.translation[2] = target[2]

    oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    # Robot handler
    pinWrapper = RobotDesigner()
    params["robot"]["end_effector_position"] = np.array(params["robot"]["end_effector_position"])
    print(params["robot"]["end_effector_position"])
    pinWrapper.initialize(params["robot"])

    print(pinWrapper.get_end_effector_frame())

    # SIMULATOR
    simulator = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        initialConfiguration=pinWrapper.get_q0_complete(),
        robotJointNames=pinWrapper.get_rmodel_complete().names,
        controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
        toolPlacement=pinWrapper.get_end_effector_frame(),
        targetPlacement=oMtarget,
        enableGUI=True,
    )

    # CONTROLLERS
    # RL Posture controller
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

    # MPC
    mpc = MPController(pinWrapper, pinWrapper.get_x0(), target, params["OCP"])

    # RICCATI
    riccati = RiccatiController()


if __name__ == "__main__":
    main()
