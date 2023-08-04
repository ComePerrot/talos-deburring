import argparse
import datetime
import pathlib
import numpy as np
import torch
import yaml

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .envs.env_talos_mpc_deburring import EnvTalosMPC
from .utils.loader_and_saver import setup_model
from .utils.custom_callbacks import (
    LoggerCallback,
    SaveFilesCallback,
    EvalOnTrainingCallback,
)

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

params_env = params["environment"]
params_designer = params["robot_designer"]
params_OCP = params["OCP"]
params_model = params["SAC"]
params_training = params["training"]


model_class = SAC

#   training parameters
number_environments = params_env["nb_environment"]
check_freq = params_training["check_freq"]
total_timesteps = params_training["total_timesteps"]
log_interval = params_training["log_interval"]

#   OCP parameters
params_OCP["state_weights"] = np.array(params_OCP["state_weights"])
params_OCP["control_weights"] = np.array(params_OCP["control_weights"])


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
if number_environments == 1:
    env_training = Monitor(
        EnvTalosMPC(params_env, params_designer, params_OCP, GUI=False),
    )
else:
    env_training = SubprocVecEnv(
        number_environments
        * [
            lambda: Monitor(
                EnvTalosMPC(params_env, params_designer, params_OCP, GUI=False),
            ),
        ],
    )

# Agent
model = setup_model(
    model_class=model_class,
    model_params=params_model,
    env_training=env_training,
)

# Callbacks
eval_callback = EvalOnTrainingCallback(
    env_training,
    eval_freq=check_freq,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)
save_files_callback = SaveFilesCallback(
    config_filename=config_filename,
    training_name=training_name,
)

callback_list = CallbackList(
    [
        save_files_callback,
        # eval_callback,
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
