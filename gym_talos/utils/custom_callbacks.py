import shutil
import time
import datetime
import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.utils import safe_mean
from typing import Optional

from gym_talos.utils.loader_and_saver import saver
from .create_target import TargetGoal


class AllCallbacks(BaseCallback):
    def __init__(
        self,
        config_filename: Optional[str] = None,
        training_name: Optional[str] = None,
        stats_window_size: int = 100,
        check_freq: int = 1000,
        verbose: int = 0,
        total_timesteps: int = 1000000,
        env: gym.Env = None,
    ):
        super().__init__(verbose)
        self.time = time.time()
        self.check_freq = check_freq
        self.config_filename = config_filename
        self.training_name = training_name
        self.best_mean_reward = -np.inf
        self._stats_window_size = stats_window_size
        self.env = env
        self.eval_on_training = None
        self._custom_info_buffer = None
        self._episode_num = 0
        self._ep_info_buffer = None
        self.save_path = None
        self._ep_end_buffer = None
        self._ep_dst_min_buffer = None
        self._dst_min = None
        self.num_timesteps_left = total_timesteps

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.save_path = self.locals["self"].logger.dir
        shutil.copy(
            self.config_filename,
            self.save_path + "/" + self.training_name + ".yaml",
        )
        # In order to have a multiple of the log_interval as check_freq
        temp = self.check_freq // self.locals["log_interval"]
        self.check_freq = temp * self.locals["log_interval"]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._update_info_buffer(self.locals["infos"][0])
        self._dst_min = (
            self.locals["infos"][0]["dst"]
            if self._dst_min is None
            else min(self._dst_min, self.locals["infos"][0]["dst"])
        )
        if self.locals["dones"][0]:
            self._episode_num += 1
            self._ep_end_buffer.extend([self.locals["infos"][0]["dst"]])
            self._ep_dst_min_buffer.extend([self._dst_min])
            self._dst_min = None
            if (
                self.locals["log_interval"] is not None
                and self._episode_num % self.locals["log_interval"] == 0
            ):
                self._dump_logs()

    def _on_training_end(self) -> None:
        saver(self.training_name, self.locals["self"])
        return True

    def _eval_training_record(self):
        if self.eval_on_training is None:
            self.eval_on_training = EvalOnTraining(
                model=self.locals["self"],
                eval_env=self.env,
                n_eval_episodes=100,
            )
        (
            eval_reward,
            eval_min_dt,
            eval_final_dt,
            eval_torque,
            eval_success,
        ) = self.eval_on_training.eval_on_train()
        if eval_reward > self.best_mean_reward:
            self.best_mean_reward = eval_reward
            self.model.save(self.save_path + "/" + "best_model.zip")
        self.logger.record("sampled_eval/reward", eval_reward)
        self.logger.record("sampled_eval/min_dt", eval_min_dt)
        self.logger.record("sampled_eval/final_dt", eval_final_dt)
        self.logger.record("sampled_eval/torque", eval_torque)
        self.logger.record("sampled_eval/success", eval_success)
        self.time = time.time()

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_per_timestep = (time.time() - self.time) / self.locals["log_interval"]
        self.num_timesteps_left -= self.locals["log_interval"]
        self.time = time.time()
        if len(self._custom_info_buffer) > 0:
            self.logger.record(
                "z_custom/torque_mean",
                safe_mean([ep_info["torque"] for ep_info in self._custom_info_buffer]),
            )
            self.logger.record(
                "z_custom/final_dt",
                safe_mean(list(self._ep_end_buffer)),
            )
            self.logger.record(
                "z_custom/min_dt",
                safe_mean(list(self._ep_dst_min_buffer)),
            )
            self.logger.record(
                ">>>     ETA    <<<",
                datetime.timedelta(
                    seconds=int(self.num_timesteps_left * time_per_timestep),
                ),
            )
            if self._episode_num % self.check_freq == 0:
                self._eval_training_record()
        return True

    def _update_info_buffer(self, infos):
        """
        Update the buffer for episode infos.
        :param infos: ([dict]) List of infos
        """
        temp_dict = {
            "torque": infos["tor"],
            "to_reach": infos["dst"],
            "from_init": infos["init"],
        }
        if self._custom_info_buffer is None:
            self._custom_info_buffer = deque(maxlen=self._stats_window_size)
        if self._ep_end_buffer is None:
            self._ep_end_buffer = deque(maxlen=self.locals["log_interval"])
        if self._ep_dst_min_buffer is None:
            self._ep_dst_min_buffer = deque(maxlen=self.locals["log_interval"])
        self._custom_info_buffer.extend([temp_dict])


class EvalOnTraining:
    """
    Callback for evaluating an agent during training.

    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to evaluate the agent
    """

    def __init__(self, model, eval_env, n_eval_episodes=100):
        self.model = model
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.target_builder = TargetGoal(self.eval_env.params_env)
        self.targets = []
        self._define_targets()

    def _define_targets(self):
        for _ in range(self.n_eval_episodes):
            self.target_builder.create_target()
            self.targets.append(self.target_builder.position_target)

    def eval_on_train(self) -> None:
        """
        This method will evaluate the agent during training
        """
        eval_rewards = []
        eval_min_dt = []
        eval_torque = []
        eval_success = []
        eval_final_dt = []
        for _ in range(self.n_eval_episodes):
            obs, infos = self.eval_env.reset(options={"target": self.targets[0]})
            episode_reward = 0.0
            episode_min_dt = infos["dst"]
            episode_torque = infos["tor"]
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, infos = self.eval_env.step(action)
                if infos["dst"] < episode_min_dt:
                    episode_min_dt = infos["dst"]
                episode_reward += reward
                done = True if terminated or truncated else False
            episode_success = int(infos["is_success"])
            episode_final_dt = infos["dst"]
            eval_rewards.append(episode_reward)
            eval_min_dt.append(episode_min_dt)
            eval_torque.append(episode_torque)
            eval_success.append(episode_success)
            eval_final_dt.append(episode_final_dt)

        return (
            np.mean(eval_rewards),
            np.mean(eval_min_dt),
            np.mean(eval_final_dt),
            np.mean(eval_torque),
            np.mean(eval_success),
        )
