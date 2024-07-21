import pickle as pkl
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from deburring_mpc import RobotDesigner

from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from gym_talos.utils.create_target import TargetGoal
from simulator.bullet_Talos import TalosDeburringSimulator


def main():
    # PARAMETERS
    filename = "config/config.yaml"
    with Path.open(filename, "r") as paramFile:
        params = yaml.safe_load(paramFile)

    target_handler = TargetGoal(params["target"])
    target_handler.create_target()
    target_handler.set_target([0.6, 0.4, 1.1])
    target = target_handler.position_target

    oMtarget = pin.SE3.Identity()
    oMtarget.translation[0] = target[0]
    oMtarget.translation[1] = target[1]
    oMtarget.translation[2] = target[2]

    oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    # Robot handler
    pinWrapper = RobotDesigner()
    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"],
    )
    pinWrapper.initialize(params["robot"])

    # SIMULATOR
    simulator = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        rmodelComplete=pinWrapper.get_rmodel_complete(),
        controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
        enableGUI=False,
        dt=float(params["timeStepSimulation"]),
    )

    # CONTROLLERS
    # MPC
    mpc = MPController(pinWrapper, pinWrapper.get_x0(), target, params["OCP"])

    # RICCATI
    riccati = RiccatiController(
        state=mpc.crocoWrapper.state,
        torque=mpc.crocoWrapper.torque,
        xref=pinWrapper.get_x0(),
        riccati=mpc.crocoWrapper.gain,
    )

    # Initialization
    Time = 0
    max_time = 4000
    # Timings
    num_simulation_step = int(
        float(params["OCP"]["time_step"]) / float(params["timeStepSimulation"]),
    )

    q_arm_list = np.zeros((int(max_time / num_simulation_step), 7))
    q_dot_arm_list = np.zeros((int(max_time / num_simulation_step), 7))

    q_list = np.zeros((int(max_time / num_simulation_step), pinWrapper.get_rmodel().nq))
    q_dot_list = np.zeros(
        (int(max_time / num_simulation_step), pinWrapper.get_rmodel().nv),
    )
    torque_list = np.zeros(
        (int(max_time / num_simulation_step), pinWrapper.get_rmodel().nq - 7),
    )

    # Control loop
    while Time < max_time:
        x_measured = simulator.getRobotState()

        if Time % num_simulation_step == 0:
            t0, x0, K0 = mpc.step(x_measured)
            riccati.update_references(t0, x0, K0)

        # Extracting full state
        q = x_measured[: pinWrapper.get_rmodel().nq]
        q_dot = x_measured[pinWrapper.get_rmodel().nq :]
        q_list[int(Time / num_simulation_step)] = q
        q_dot_list[int(Time / num_simulation_step)] = q_dot
        torque_list[int(Time / num_simulation_step)] = t0

        # Extracting arm state
        q_arm = x_measured[15 + 7 : 22 + 7]
        q_dot_arm = x_measured[32 + 15 + 7 : 32 + 22 + 7]
        q_arm_list[int(Time / num_simulation_step)] = q_arm
        q_dot_arm_list[int(Time / num_simulation_step)] = q_dot_arm

        torques = riccati.step(x_measured)
        simulator.step(torques, pinWrapper.get_end_effector_frame(), oMtarget)

        Time += 1

    simulator.end()
    # with open("arm_state.pkl", "wb") as file:
    #     pkl.dump([q_arm_list, q_dot_arm_list], file)

    with Path.open("reach_movement.pkl", "wb") as file:
        pkl.dump([q_list, q_dot_list, torque_list], file)


if __name__ == "__main__":
    main()
