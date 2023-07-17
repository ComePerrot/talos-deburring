import argparse
import datetime
import pathlib
import shutil

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from deburring_mpc import OCPSettings

from .envs.env_talos_mpc_deburring import EnvTalosMPC

################
#  PARAMETERS  #
################
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-id",
    "--identication",
    default=None,
    help=(
        "Identification number for the training (usefull when launching several "
        "trainings in parallel)"
    ),
    type=int,
)
parser.add_argument(
    "-config",
    "--configurationFile",
    required=True,
    help="Path to file containg the configuration of the training",
)

args = parser.parse_args()
config_filename = pathlib.Path(args.configurationFile)
training_id = args.identication

# Parsing configuration file
with config_filename.open() as config_file:
    params = yaml.safe_load(config_file)

params_designer = params["robot_designer"]
params_env = params["environment"]
params_training = params["training"]

# parameter OCP
OCPparams = OCPSettings()
OCPparams.read_from_yaml("./config/config_MPC_RL.yaml")

# Setting names and log locations
now = datetime.datetime.now()
current_date = str(now.strftime("%Y-%m-%d"))
if training_id:
    training_name = (
        current_date + "_" + params_training["name"] + "_" + str(training_id)
    )
else:
    training_name = current_date + "_" + params_training["name"]

log_dir = "./logs/"

number_environments = params_training["environment_quantity"]
total_timesteps = params_training["total_timesteps"]
verbose = params_training["verbose"]

torch.set_num_threads(1)

##############
#  TRAINING  #
##############
# Create environment
if number_environments == 1:
    env_training = EnvTalosMPC( params_env, params_designer, OCPparams, GUI=True)
else:
    env_training = SubprocVecEnv(
        number_environments
        * [lambda: Monitor(EnvTalosMPC(params_env, params_designer,  OCPparams, GUI=False))],
    )

# Create Agent
model = SAC(
    "MlpPolicy",
    env_training,
    verbose=verbose,
    tensorboard_log=log_dir,
    device="cpu",
)

# Train Agent
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=training_name,
)

env_training.close()

# Save agent and config file
model.save(model.logger.dir + "/" + training_name)
shutil.copy(config_filename, model.logger.dir + "/" + training_name + ".yaml")