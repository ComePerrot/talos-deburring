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
    filename = "config/config.yaml"
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
    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"]
    )
    pinWrapper.initialize(params["robot"])

    # SIMULATOR
    simulator = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        initialConfiguration=pinWrapper.get_q0_complete(),
        robotJointNames=pinWrapper.get_rmodel_complete().names,
        controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
        toolPlacement=pinWrapper.get_end_effector_frame(),
        targetPlacement=oMtarget,
        enableGUI=False,
    )

    # CONTROLLERS
    #   RL Posture controller
    #       Action wrapper
    kwargs_action = dict(
        rl_controlled_IDs=[16, 17, 19],
        rmodl=pinWrapper.get_rmodel(),
        scaling_factor=1,
        scaling_mode="full_range",
        initial_pose=None,
    )
    #       Observation wrapper
    kwargs_observation = dict(
        normalize_obs=True,
        rmodel=pinWrapper.get_rmodel(),
        target_handler=target_handler,
        history_size=0,
        prediction_size=0,
    )
    model_path = "./test"
    posture_controller = RLPostureController(
        model_path, kwargs_action, kwargs_observation
    )

    # MPC
    mpc = MPController(pinWrapper, pinWrapper.get_x0(), target, params["OCP"])

    # RICCATI
    riccati = RiccatiController()


if __name__ == "__main__":
    main()
