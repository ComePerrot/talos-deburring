import pinocchio as pin
import numpy as np
import yaml

from deburring_mpc import RobotDesigner

from gym_talos.utils.create_target import TargetGoal

from simulator.bullet_Talos import TalosDeburringSimulator
from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from controllers.RL_posture import RLPostureController

from IPython import embed


def main():
    # PARAMETERS
    filename = "config/config.yaml"
    with open(filename, "r") as paramFile:
        params = yaml.safe_load(paramFile)

    # Robot handler
    pinWrapper = RobotDesigner()
    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"]
    )
    pinWrapper.initialize(params["robot"])

    joints_to_lock = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        22,
        23,
        24,
        25,
        26,
        27,
    ]

    rmodel_arm = pin.buildReducedModel(
        pinWrapper.get_rmodel(), joints_to_lock, pinWrapper.get_q0()
    )
    rdata_arm = rmodel_arm.createData()

    pin.rnea(
        rmodel_arm,
        rdata_arm,
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0]),
    )

    force = rdata_arm.f[1]

    print(np.linalg.norm(force.angular))

    embed()


if __name__ == "__main__":
    main()
