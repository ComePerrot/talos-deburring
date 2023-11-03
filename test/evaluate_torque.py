import pinocchio as pin
import numpy as np
import yaml
import pickle as pkl
import matplotlib.pyplot as plt

from deburring_mpc import RobotDesigner


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
        23,
        24,
        25,
        26,
        27,
    ]

    rmodel_arm = pin.buildReducedModel(
        pinWrapper.get_rmodel(), joints_to_lock, pinWrapper.get_q0()
    )
    rdata_arm_full = rmodel_arm.createData()
    rdata_arm_reduced = rmodel_arm.createData()

    with open("arm_state.pkl", "rb") as file:
        q_lists = pkl.load(file)
        q_arm_list = q_lists[0]
        q_dot_arm_list = q_lists[1]

    torque_full_norm_list = np.zeros(len(q_arm_list))
    torque_reduced_norm_list = np.zeros(len(q_arm_list))

    for i in range(len(q_arm_list)):
        q_arm = q_arm_list[i]
        q_dot_arm = q_dot_arm_list[i]
        q_ddot_arm = np.diff(q_dot_arm)
        q_ddot_arm = np.concatenate([[0], q_ddot_arm])

        pin.rnea(
            rmodel_arm,
            rdata_arm_full,
            q_arm,
            q_dot_arm,
            q_ddot_arm,
        )
        force_full = rdata_arm_full.f[1]
        torque_full = force_full.angular
        pin.rnea(
            rmodel_arm,
            rdata_arm_reduced,
            q_arm,
            np.zeros(7),
            np.zeros(7),
        )
        force_reduced = rdata_arm_reduced.f[1]
        torque_reduced = force_reduced.angular

        torque_full_norm_list[i] = np.linalg.norm(torque_full)
        torque_reduced_norm_list[i] = np.linalg.norm(torque_reduced)

    plt.plot(torque_full_norm_list, label="Full torque")
    plt.plot(torque_reduced_norm_list, label="Reduced torque")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
