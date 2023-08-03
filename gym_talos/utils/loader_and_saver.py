from typing import Optional
import gymnasium
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer


def saver(
    training_name: str,
    model: BaseAlgorithm,
) -> None:
    """
    This function saves the model

    :param training_name: The name of the training
    :param model: The model to be saved
    """
    print("Saving model as {}".format(model.logger.dir + "/" + training_name))
    model.save(model.logger.dir + "/" + training_name)


def handler(
    signum,
    frame,
    training_name: str,
    model: BaseAlgorithm,
) -> None:
    """
    This function is called when CTRL+C is pressed

    The goal of this callback is to save an early training end

    :param signum: The signal number
    :param frame: The current stack frame
    :param training_name: The name of the training
    :param model: The model to be saved
    """
    saver(training_name=training_name, model=model)
    exit(1)


def setup_env(
    env_class: gymnasium.Env,
    env_params: dict,
    designer_params: dict,
    GUI: bool = False,
) -> gymnasium.Env:
    """
    This function creates the environment used for training

    :param env_class: The class of the environment to be used
    :param env_params: The parameters of the environment saved in a yaml file
    :param designer_params: The parameters of the designer saved in a yaml file
    :param GUI: Whether or not to display the GUI
    :return: The environment created
    """
    if env_params["nb_environments"] == 1:
        return Monitor(env_class(designer_params, env_params, GUI=GUI))
    return SubprocVecEnv(
        env_params["nb_environments"]
        * [lambda: Monitor(env_class(designer_params, env_params, GUI=False))],
    )


def setup_model(
    model_class: BaseAlgorithm,
    model_params: dict,
    env_training: gymnasium.Env,
    replay_buffer_class: Optional[DictReplayBuffer] = None,
) -> BaseAlgorithm:
    """
    This function creates the model used for training

    :param model_class: The class of the model to be used
    :param model_params: The parameters of the model saved in a yaml file
    :param env_training: The environment used for training
    :param replay_buffer_class: The class of the replay buffer to be used
    :return: The model created
    """
    if type(replay_buffer_class) == type(HerReplayBuffer):
        model_params = {
            **{
                "replay_buffer_class": replay_buffer_class,
                "env": env_training,
            },
            **model_params["model_param"],
            **model_params["HerReplayBuffer_param"],
        }
    else:
        model_params = {
            **{
                "env": env_training,
            }
            ** model_params["model_param"],
        }
    return model_class(**model_params)
