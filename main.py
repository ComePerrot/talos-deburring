from pathlib import Path
import numpy as np
import yaml

from deburring_mpc import RobotDesigner

from gym_talos.utils.create_target import TargetGoal

from simulator.bullet_Talos import TalosDeburringSimulator
from simulator.mujoco_Talos import TalosMujoco
from factory.benchmark_MPRL import bench_MPRL
from factory.benchmark_MPC import bench_MPC
from factory.benchmark_MPC_variablePosture import bench_MPC_variablePosture
from factory.benchmark_MPC_noRiccati import bench_MPC_noRiccati


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

    # Robot handler
    pinWrapper = RobotDesigner()
    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"]
    )
    pinWrapper.initialize(params["robot"])

    # SIMULATOR
    simulator = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        rmodelComplete=pinWrapper.get_rmodel_complete(),
        controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
        enableGUI=params["GUI"],
        dt=float(params["timeStepSimulation"]),
        cutoff_frequency=params["robot_cutoff_frequency"],
    )

    MPRL = bench_MPRL(parameter_file, target_handler, pinWrapper, simulator)
    MPC = bench_MPC(parameter_file, pinWrapper, simulator)
    MPC_variablePosture = bench_MPC_variablePosture(parameter_file, pinWrapper, simulator)
    MPC_noRiccati = bench_MPC_noRiccati(parameter_file, pinWrapper, simulator)

    for controller in [MPC]:
        print(type(controller).__name__)
        catastrophic_failure = 0
        failure = 0
        success = 0
        for target in targets:
            (
                reach_time,
                error_placement_tool,
                limits,
            ) = controller.run(target)

            if verbose == 1:
                print(np.array(target))
            if limits is not False:
                catastrophic_failure += 1

                if verbose == 1:
                    print("Limit infriged")
                    print(limits)
            else:
                if verbose == 1:
                    print(reach_time, error_placement_tool)
                if reach_time is not None:
                    success += 1
                else:
                    failure += 1
        print("Catastrophic failure:")
        print(" " + str(catastrophic_failure))
        print("Failure: " + str(failure))
        print("Success: " + str(success))

    simulator.end()


if __name__ == "__main__":
    main()
