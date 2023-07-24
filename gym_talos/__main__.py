import argparse
import datetime
import pathlib
import signal

import torch
import yaml
from stable_baselines3 import HerReplayBuffer, SAC

from .envs.env_talos_deburring_her import EnvTalosDeburringHer

from .utils.custom_callbacks import AllCallbacks
from .utils.loader_and_saver import saver, handler, setup_model, setup_env

################
# Main HER SAC #
################

# Need to add also for the goal the speed of the ARM probably,
# cause need to stop the movement at the right time
# However with HER program, the issue is that it would mean that every time we need to
# stop the movement at the right time
# And so replay buffers arent good for that since the movement is stopped earlier than
# the goal, so with speed ?

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

designer_params = params["robot_designer"]
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

total_timesteps = params_training["total_timesteps"]
log_interval = params_training["log_interval"]

torch.set_num_threads(1)


################
#  PARAMETERS  #
################

env_class = EnvTalosDeburringHer
env_params = params["environment"]

model_class = SAC
model_params = params["SAC"]

replay_buffer_class = HerReplayBuffer

env_training = setup_env(
    env_class=env_class,
    env_params=env_params,
    designer_params=designer_params,
)
model = setup_model(
    model_class=model_class,
    model_params=model_params,
    env_training=env_training,
    replay_buffer_class=replay_buffer_class,
)

callback_class = AllCallbacks(
    config_filename=config_filename,
    training_name=training_name,
    stats_window_size=100,
    check_freq=1000,
    verbose=1,
    env=env_training,
)
# Callback function to save the model when CTRL+C is pressed
signal.signal(
    signal.SIGINT,
    lambda signum, frame: handler(
        signum=signum,
        frame=frame,
        training_name=training_name,
        model=model,
    ),
)

# Train Agent

model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=training_name,
    log_interval=log_interval,
    callback=callback_class,
)

saver(training_name=training_name, model=model)
env_training.close()
