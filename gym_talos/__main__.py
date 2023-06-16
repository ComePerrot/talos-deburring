import argparse
import datetime
import pathlib
import shutil

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .envs.env_talos_deburring import EnvTalosDeburring

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
learning_rate = params_training["learning_rate"]
train_freq = params_training["train_freq"]
learning_starts = params_training["learning_starts"]
log_interval = params_training["log_interval"]

torch.set_num_threads(1)

##############
#  TRAINING  #
##############
# Create environment
if number_environments == 1:
    env_training = EnvTalosDeburring(params_designer, params_env, GUI=False)
else:
    env_training = SubprocVecEnv(
        number_environments
        * [lambda: Monitor(EnvTalosDeburring(params_designer, params_env, GUI=False))],
    )

# Create Agent
model = SAC(
    "MlpPolicy",
    env_training,
    verbose=verbose,
    tensorboard_log=log_dir,
    learning_rate=learning_rate,
    train_freq=train_freq,
    learning_starts=learning_starts,
    device="cpu",
)

# Train Agent
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=training_name,
    log_interval=log_interval,
)
target_pos = env_training.targetPos
env_training.close()

# Save agent and config file
model.save(model.logger.dir + "/" + training_name)
f = open(model.logger.dir + "/" + training_name + ".txt", "a")
f.write("Position: " + str(target_pos) + "\n")
f.close()
shutil.copy(config_filename, model.logger.dir + "/" + training_name + ".yaml")
