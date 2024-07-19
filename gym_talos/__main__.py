import argparse
import datetime
import pathlib
import shutil
import signal
import os

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from .envs.env_talos_deburring_her import EnvTalosDeburringHer
from .utils.tb_callback import AllCallbacks, SaveCallback, TensorboardCallback

################
# Main HER SAC #
################

# Need to add also for the goal the speed of the ARM probably, cause need to stop the movement at the right time
# However with HER program, the issue is that it would mean that every time we need to stop the movement at the right time
# And so replay buffers arent good for that since the movement is stopped earlier than the goal, so with speed ? 

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
gamma = params_training["gamma"]
batch_size = params_training["batch_size"]
learning_rate = params_training["learning_rate"]
buffer_size = int(float(params_training["buffer_size"]))
tau = params_training["tau"]
log_interval = params_training["log_interval"]
n_sampled_goal = params_training["n_sampled_goal"]
learning_starts = params_training["learning_starts"]

torch.set_num_threads(1)



##############
#  TRAINING  #
##############
# Create environment
env_class = EnvTalosDeburringHer
model_class = SAC  # works also with SAC, DDPG and TD3
callback_class = AllCallbacks(config_filename=config_filename, 
                            training_name=training_name,
                            stats_window_size=100, 
                            check_freq=1000, 
                            verbose=verbose)
save_callback = SaveCallback(config_filename=config_filename, 
                            training_name=training_name,
                            check_freq=1000,
                            verbose=verbose
                            )
tensorboard_callback = TensorboardCallback(stats_window_size=100,
                                           verbose=verbose)
callback_list = CallbackList([save_callback, tensorboard_callback])

if number_environments == 1:
    env_training = env_class(params_designer, params_env, GUI=False)
else:
    env_training = SubprocVecEnv(
        number_environments
        * [lambda: Monitor(env_class(params_designer, params_env, GUI=False))],
    )

goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE
model = model_class(
    "MultiInputPolicy",
    env_training,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=verbose,
    learning_starts=learning_starts,
    tensorboard_log=log_dir,
    device="cpu",
    buffer_size=buffer_size,
    learning_rate=learning_rate,
    gamma=gamma, batch_size=batch_size, tau=tau,
    policy_kwargs=dict(net_arch=[512, 512, 512])
)

def saver(training_name, model):
    print("Saving model as {}".format(model.logger.dir + "/" + training_name))
    model.save(model.logger.dir + "/" + training_name)

def handler(signum, frame):
    saver(training_name=training_name, 
          model=model)
    exit(1)

signal.signal(signal.SIGINT, handler)

# Train Agent

model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=training_name,
    log_interval=log_interval,
    callback=callback_class,
)

saver(training_name=training_name, 
      model=model)
env_training.close()