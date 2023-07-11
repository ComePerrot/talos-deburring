import yaml
import os

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator
from gym_talos.utils.modelLoader import TalosDesigner
from deburring_mpc import OCPSettings

from stable_baselines3.common.env_checker import check_env
from .envs.env_talos_mpc_deburring import EnvTalosMPC

# Parsing configuration file
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = "/../config/config_MPC_RL.yaml"
config_file = dir_path + filename
with open(config_file) as config_file:
    params = yaml.safe_load(config_file)

# parameter OCP
OCPparams = OCPSettings()
OCPparams.read_from_yaml("./config/config_MPC_RL.yaml")

params_designer = params["robot_designer"]
params_env = params["environment"]
params_training = params["training"]


check_env(EnvTalosMPC(params_env, params_designer, OCPparams))

# pinWrapper = TalosDesigner(
#     URDF="/talos_data/robots/talos_reduced.urdf",
#     SRDF="/talos_data/srdf/talos.srdf",
#     toolPosition=[0, -0.02, -0.0825],
#     controlledJoints=[
#         "arm_left_1_joint",
#         "arm_left_2_joint",
#         "arm_left_3_joint",
#         "arm_left_4_joint",
#     ],
# )

# rmodel = pinWrapper.rmodel

# simulator = TalosDeburringSimulator(
#     URDF=pinWrapper.URDF_path,
#     rmodelComplete=pinWrapper.rmodelComplete,
#     controlledJointsIDs=pinWrapper.controlledJointsID,
#     enableGUI=True,
#     dt=10,
# )

# while True:
#     simulator.step()
#     pass
