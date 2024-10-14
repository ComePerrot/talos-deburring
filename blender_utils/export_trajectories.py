import pickle
from pathlib import Path

import numpy as np
import yaml
from deburring_benchmark.factory.benchmark_MPC import bench_MPC
from robot_description.path_getter import srdf_path, urdf_path
from simulator.bullet_Talos import TalosDeburringSimulator

from deburring_mpc import RobotDesigner


def setup_parameters():
    filepath = Path(__file__).resolve().parent
    parameter_file = (
        filepath / "../deburring-python-utils/deburring_benchmark/config_benchmark.yaml"
    )
    with parameter_file.open(mode="r") as paramFile:
        params = yaml.safe_load(paramFile)

    params["robot"]["urdf_path"] = urdf_path[params["robot"]["urdf_type"]]
    params["robot"]["srdf_path"] = srdf_path

    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"],
    )

    return params


def export_joint_data(file_path, x_list):
    with Path.open(file_path, "wb") as file:
        pickle.dump(x_list, file)


if __name__ == "__main__":
    # Parameters
    params = setup_parameters()
    params["logging_frequency"] = 60  # Hz
    target = [0.6, 0.4, 1.1]

    # Robot handler
    pinWrapper = RobotDesigner()
    pinWrapper.initialize(params["robot"])

    # Simulator
    simulator = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        rmodel_complete=pinWrapper.get_rmodel_complete(),
        controlled_joints_ids=pinWrapper.get_controlled_joints_ids(),
        enable_GUI=params["GUI"],
        dt=float(params["timeStepSimulation"]),
        cutoff_frequency=params["robot_cutoff_frequency"],
    )

    # Baseline trajectory (Vanilla MPC)
    bench_MPC = bench_MPC(params, pinWrapper, simulator, logging=True)
    bench_MPC.run(target)

    filepath = Path(__file__).resolve().parent
    export_joint_data(filepath / "trajectories/trajectory_MPC.pkl", bench_MPC.x_log)
