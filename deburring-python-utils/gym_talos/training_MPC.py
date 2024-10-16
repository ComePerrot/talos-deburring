import argparse
import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

from gym_talos.envs.env_talos_mpc_deburring import EnvTalosMPC
from gym_talos.utils.custom_callbacks import (
    EvalOnTrainingCallback,
    LoggerCallback,
    SaveFilesCallback,
)
from gym_talos.utils.loader_and_saver import setup_env, setup_model

################
#  PARAMETERS  #
################

env_class = EnvTalosMPC
model_class = SAC

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-id",
    "--identification",
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
    default=(Path(__file__).resolve().parent / "config/config_MPC_RL.yaml"),
    required=False,
    help="Path to file containg the configuration of the training",
)

args = parser.parse_args()
config_filename = Path(args.configurationFile)
print(config_filename)
training_id = args.identification

# Parsing configuration file
with config_filename.open() as config_file:
    params = yaml.safe_load(config_file)

params_robot = params["robot"]
params_env = params["environment"]
params_model = params["SAC"]
params_training = params["training"]

#   training parameters
check_freq = params_training["check_freq"]
total_timesteps = params_training["total_timesteps"]
log_interval = params_training["log_interval"]
number_eval = params_training["number_eval"]

# Setting names and log locations
now = datetime.datetime.now()
current_date = str(now.strftime("%Y-%m-%d"))
if training_id:
    training_name = (
        current_date + "_" + params_training["name"] + "_" + str(training_id)
    )
else:
    training_name = current_date + "_" + params_training["name"]

torch.set_num_threads(1)

##############
#  TRAINING  #
##############
# Environment
env_training = setup_env(
    env_class=env_class,
    env_params=params_env,
    designer_params=params_robot,
    GUI=False,
)

# Agent
model = setup_model(
    model_class=model_class,
    model_params=params_model,
    env_training=env_training,
)

# Callbacks
save_files_callback = SaveFilesCallback(
    config_filename=config_filename,
    training_name=training_name,
)
eval_callback = EvalOnTrainingCallback(
    env_training,
    eval_freq=check_freq,
    n_eval_episodes=number_eval,
    deterministic=True,
    render=False,
)

callback_list = CallbackList(
    [
        save_files_callback,
        eval_callback,
    ],
)

# Train Agent
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=training_name,
    log_interval=log_interval,
    callback=callback_list,
)
env_training.close()
