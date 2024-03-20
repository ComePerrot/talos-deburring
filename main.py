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


def setup_benchmark(parameter_file):
    """Set up the benchmark.

    Set up the benchmark by loading parameters, creating a target handler,
    initializing the robot designer, and setting up the simulator.

    Args:
        parameter_file (Path): The path to the parameter file in YAML format.
    """
    with parameter_file.open(mode="r") as paramFile:
        params = yaml.safe_load(paramFile)

    target_handler = TargetGoal(params["target"])
    target_handler.create_target()

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

    return (params, target_handler, pinWrapper, simulator)


def run_benchmark(targets, trial_list):
    """Run the benchmark.

    Executes each trial for each target and stores the results.

    Args:
        targets: A list of targets for the benchmark.
        trial_list: A list of controllers to be tested in the benchmark.
    """
    results = BenchmarkResult()
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

        results.add_trial_result(
            trial_id,
            type(controller).__name__,
            test_details,
        )

    # Closing simulator (all trials are talking to the same instance)
    trial_list[0].simulator.end()

    return results


if __name__ == "__main__":
    # Parameters
    parameter_filename = "config/config.yaml"

    test_MPC = False
    test_MPC_variablePosture = False
    rl_model_paths = [
        "/home/cperrot/ws_bench/logs/2024-03-20_3joints_2/best_model",
    ]

    # Loading parameters from YAML file
    filepath = Path(__file__).resolve().parent
    parameter_file = filepath / parameter_filename

    # Setting up useful objects
    params, target_handler, pinWrapper, simulator = setup_benchmark(parameter_file)

    # Creating trial list
    trial_list = []
    if test_MPC:
        MPC = bench_MPC(params, pinWrapper, simulator)
        trial_list.append(MPC)

    if test_MPC_variablePosture:
        MPC_variablePosture = bench_MPC_variablePosture(params, pinWrapper, simulator)
        trial_list.append(MPC_variablePosture)

    if rl_model_paths is not None:
        MPRL_list = []
        for path in rl_model_paths:
            MPRL = bench_MPRL(params, path, target_handler, pinWrapper, simulator)
            MPRL_list.append(MPRL)
        trial_list.extend(MPRL_list)

    # Running benchmark
    targets = target_handler.generate_target_list(params["numberTargets"])
    results = run_benchmark(targets, trial_list)

    # Printing results
    results.print_results()
