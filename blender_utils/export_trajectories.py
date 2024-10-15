import pickle
from pathlib import Path

import numpy as np
import yaml
from deburring_benchmark.factory.benchmark_MPC import bench_MPC
from deburring_benchmark.factory.benchmark_MPC_variablePosture import (
    bench_MPC_variablePosture,
)
from deburring_benchmark.factory.benchmark_MPRL import bench_MPRL
from gym_talos.utils.create_target import TargetGoal
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
    target = [0.9 , 0.15, 1.145]

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

    filepath = Path(__file__).resolve().parent

    # Baseline trajectory (Vanilla MPC)
    print("Running MPC benchmark")
    benchmark_MPC = bench_MPC(params, pinWrapper, simulator, logging=True)
    _, error_placement_tool, _ = benchmark_MPC.run(target)
    print(f"Placement error: {error_placement_tool * 1000:.1f} mm")

    # Variable posture MPC
    print("Running variable posture MPC benchmark")
    benchmark_variablePosture = bench_MPC_variablePosture(
        params,
        pinWrapper,
        simulator,
        logging=True,
    )
    _, error_placement_tool, _ = benchmark_variablePosture.run(target)
    print(f"Placement error: {error_placement_tool * 1000:.1f} mm")

    # MP/RL
    print("Running MPC/RL benchmark")
    target_handler = TargetGoal(params["target"])
    target_handler.create_target()
    policy_full_path = (
        filepath / "../deburring-python-utils/deburring_benchmark/example_policy"
    )
    benchmark_MPRL = bench_MPRL(
        params,
        policy_full_path,
        target_handler,
        pinWrapper,
        simulator,
        logging=True,
    )
    _, error_placement_tool, _ = benchmark_MPRL.run(target)
    print(f"Placement error: {error_placement_tool * 1000:.1f} mm")

    print("Exporting trajectories")
    export_joint_data(filepath / "trajectories/trajectory_MPC.pkl", benchmark_MPC.x_log)
    export_joint_data(
        filepath / "trajectories/trajectory_varPostMPC.pkl",
        benchmark_variablePosture.x_log,
    )
    export_joint_data(
        filepath / "trajectories/trajectory_MPRL.pkl",
        benchmark_MPRL.x_log,
    )
    print("Done")
