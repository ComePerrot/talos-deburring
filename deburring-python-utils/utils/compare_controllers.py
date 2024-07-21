import numpy as np
import yaml
import pickle as pkl
from stable_baselines3 import SAC

from deburring_mpc import RobotDesigner

from gym_talos.utils.create_target import TargetGoal
from gym_talos.envs.env_talos_mpc_deburring import EnvTalosMPC

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator

# from simulator.bullet_Talos import TalosDeburringSimulator
from factory.benchmark_MPRL import bench_MPRL
from factory.benchmark_MPC import bench_MPC


target = [0.5, 0.3, 1.05]
test_type = "gym" # "gym" or "bench" or "bench with gym"

print(test_type)

def run_gym(target):
    obs, infos = envDisplay.reset(options={"target": target})
    print(envDisplay.target_handler.position_target)
    done = False
    i = 0
    while not done:
        i += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, infos = envDisplay.step(
            action,
        )
        if truncated:
            done = True
            print("Truncated")
        elif terminated:
            time = infos["time"]
            distance = infos["dst"]

            if distance < 0.005:
                print(float(time) * 0.2, distance)
            else:
                print(time, distance)

            done = True
    envDisplay.close()


if test_type == "gym":
    model_path = "config/best_model.zip"
    filename_gym = "config/config_gym.yaml"
    with open(filename_gym, "r") as parameter_file:
        params_gym = yaml.safe_load(parameter_file)

    envDisplay = EnvTalosMPC(
        params_gym["robot"],
        params_gym["environment"],
        GUI=False,
    )

    model = SAC.load(model_path, env=envDisplay)
    
    run_gym(target)

    with open("gym_data.pkl", "wb") as file:
        pkl.dump([envDisplay.x_list, envDisplay.u_list, envDisplay.xref_list], file)

elif test_type == "bench":
    filename_bench = "config/config.yaml"
    with open(filename_bench, "r") as parameter_file:
        params_bench = yaml.safe_load(parameter_file)
    
    target_handler = TargetGoal(params_bench["target"])
    target_handler.create_target()

    pinWrapper = RobotDesigner()
    params_bench["robot"]["end_effector_position"] = np.array(
        params_bench["robot"]["end_effector_position"]
    )
    pinWrapper.initialize(params_bench["robot"])

    simulator_bench = TalosDeburringSimulator(
        URDF=pinWrapper.get_settings()["urdf_path"],
        rmodelComplete=pinWrapper.get_rmodel_complete(),
        controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
        enableGUI=False,
        dt=float(params_bench["timeStepSimulation"]),
    )

    MPRL = bench_MPRL(filename_bench, target_handler, pinWrapper, simulator_bench)
    MPC = bench_MPC(filename_bench, pinWrapper, simulator_bench)
    (
        reach_time,
        error_placement_tool,
        limit_position,
        limit_speed,
        limit_command,
    ) = MPRL.run(target)

    if limit_position or limit_speed:
        if limit_position:
            print("Position limit infriged")
        elif limit_speed:
            print("Speed limit infriged")
    else:
        print(reach_time, error_placement_tool)

    with open("bench_data.pkl", "wb") as file:
        pkl.dump([MPRL.x_list, MPRL.u_list, MPRL.xref_list], file)

elif test_type == "bench with gym":
    filename_bench = "config/config.yaml"
    
    filename_gym = "config/config_gym.yaml"
    with open(filename_gym, "r") as parameter_file:
        params_gym = yaml.safe_load(parameter_file)

    envDisplay = EnvTalosMPC(
        params_gym["robot"],
        params_gym["environment"],
        GUI=False,
    )

    MPC = bench_MPC(filename_bench, envDisplay.pinWrapper, envDisplay.simulator)
    (
        reach_time,
        error_placement_tool,
        limit_position,
        limit_speed,
        limit_command,
    ) = MPC.run(target)

    if limit_position or limit_speed:
        if limit_position:
            print("Position limit infriged")
        elif limit_speed:
            print("Speed limit infriged")
    else:
        print(reach_time, error_placement_tool)
else:
    msg = "Unknown test type"
    raise ValueError(msg)

# print("first state = " + str(x0))
# print("first torque = " + str(u0))

# ddp_bench = MPC.mpc.crocoWrapper.solver
# ddp_gym = envDisplay.crocoWrapper.solver

# xs_bench_list = ddp_bench.xs.tolist()
# xs_gym_list = ddp_gym.xs.tolist()
# for xs_bench, xs_gym in zip(xs_bench_list, xs_gym_list):
#     if (xs_bench!=xs_gym).any():
#         print('NON')

# us_bench_list = ddp_bench.us.tolist()
# us_gym_list = ddp_gym.us.tolist()

# for us_bench, us_gym in zip(us_bench_list, us_gym_list):
#     if (us_bench!=us_gym).any():
#         print('NON')

# K_bench_list = ddp_bench.K.tolist()
# K_gym_list = ddp_gym.K.tolist()
# for K_bench, K_gym in zip(K_bench_list, K_gym_list):
#     if (K_bench!=K_gym).any():
#         print('NON')

# for runningModel_bench, runningModel_gym in zip(
#     ddp_bench.problem.runningDatas, ddp_gym.problem.runningDatas
# ):
#     costs_bench = runningModel_bench.differential.costs.costs.todict()
#     costs_gym = runningModel_gym.differential.costs.costs.todict()