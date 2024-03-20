from pathlib import Path
import numpy as np
import yaml

from deburring_mpc import RobotDesigner

from gym_talos.utils.create_target import TargetGoal

from robot_description.path_getter import urdf_path, srdf_path

from simulator.bullet_Talos import TalosDeburringSimulator
from factory.benchmark_MPRL import bench_MPRL
from factory.benchmark_MPC import bench_MPC
from factory.benchmark_MPC_variablePosture import bench_MPC_variablePosture
from factory.benchmark_results import BenchmarkResult


def main():
    # PARAMETERS
    parameter_filename = "config/config.yaml"
    filepath = Path(__file__).resolve().parent
    parameter_file = filepath / parameter_filename
    with parameter_file.open(mode="r") as paramFile:
        params = yaml.safe_load(paramFile)

    verbose = params["verbose"]
    target_handler = TargetGoal(params["target"])
    target_handler.create_target()
    targets = target_handler.generate_target_list(params["numberTargets"])

    params["robot"]["urdf_path"] = urdf_path["example_robot_data"]
    params["robot"]["srdf_path"] = srdf_path

    # Robot handler
    pinWrapper = RobotDesigner()
    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"],
    )
    pinWrapper.initialize(params["robot"])

    # SIMULATOR
    simulator = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        rmodel_complete=pinWrapper.get_rmodel_complete(),
        controlled_joints_ids=pinWrapper.get_controlled_joints_ids(),
        enable_GUI=params["GUI"],
        dt=float(params["timeStepSimulation"]),
        cutoff_frequency=params["robot_cutoff_frequency"],
    )

    MPC = bench_MPC(params, pinWrapper, simulator)
    MPC_variablePosture = bench_MPC_variablePosture(
        params,
        pinWrapper,
        simulator,
    )
    # MPRL
    model_paths = [
        "/home/cperrot/ws_bench/logs/2024-03-20_3joints_2/best_model",
    ]
    MPRL_list = []
    for path in model_paths:
        MPRL = bench_MPRL(params, path, target_handler, pinWrapper, simulator)
        MPRL_list.append(MPRL)

    trial_list = [*MPRL_list]
    result = BenchmarkResult()
    for trial_id, controller in enumerate(trial_list):
        test_details = []
        for target in targets:
            (
                reach_time,
                error_placement_tool,
                limits,
            ) = controller.run(target)

            test_detail = {
                "target": target,
                "reach_time": reach_time,
                "error_placement_tool": error_placement_tool,
                "limits": limits,
            }
            test_details.append(test_detail)

        result.add_trial_result(
            trial_id,
            type(controller).__name__,
            test_details,
        )

    result.print_results()
    simulator.end()


if __name__ == "__main__":
    main()
