import numpy as np
import yaml

from deburring_mpc import RobotDesigner

from gym_talos.utils.create_target import TargetGoal

from simulator.bullet_Talos import TalosDeburringSimulator
from factory.benchmark_MPRL import bench_MPRL
from factory.benchmark_MPC import bench_MPC

def main():
    # PARAMETERS
    filename = "config/config.yaml"
    with open(filename, "r") as paramFile:
        params = yaml.safe_load(paramFile)

    target_handler = TargetGoal(params["target"])
    target_handler.create_target()
    targets = target_handler.generate_target_list([3,3,3])

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
        enableGUI=False,
        dt=float(params["timeStepSimulation"]),
    )

    MPRL = bench_MPRL(filename, target_handler, pinWrapper, simulator)
    MPC = bench_MPC(filename, pinWrapper, simulator)

    for controller in [MPC, MPRL]:
        for target in targets:
            (
                reach_time,
                error_placement_tool,
                limit_position,
                limit_speed,
                limit_command,
            ) = controller.run(target)

            print(target)
            if (limit_position or limit_speed or limit_command):
                if limit_position:
                    print("Position limit infriged")
                elif limit_speed:
                    print("Speed limit infriged")
                else:
                    print("Control limit infriged")
            else:
                print(reach_time, error_placement_tool)

    simulator.end


if __name__ == "__main__":
    main()
