import shutil
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.utils import safe_mean
from typing import Optional


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, stats_window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._stats_window_size = stats_window_size
        self._custom_info_buffer = None
        self._episode_num = 0

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._update_info_buffer(self.locals["infos"][0])
        if self.locals["dones"][0]:
            self._episode_num += 1
            if (
                self.locals["log_interval"] is not None
                and self._episode_num % self.locals["log_interval"] == 0
            ):
                self._dump_logs()

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        if len(self._custom_info_buffer) > 0:
            self.logger.record(
                "z_custom/torque_mean",
                safe_mean([ep_info["torque"] for ep_info in self._custom_info_buffer]),
            )
            self.logger.record(
                "z_custom/to_reach_mean",
                safe_mean(
                    [ep_info["to_reach"] for ep_info in self._custom_info_buffer],
                ),
            )
            self.logger.record(
                "z_custom/from_init_mean",
                safe_mean(
                    [ep_info["from_init"] for ep_info in self._custom_info_buffer],
                ),
            )
        ## If we want saves more frequently we can with the following command
        # self.logger.dump(step=self.num_timesteps)
        return True

    def _update_info_buffer(self, infos):
        """
        Update the buffer for episode infos.
        :param infos: ([dict]) List of infos
        """
        temp_dict = {}
        if self._custom_info_buffer is None:
            self._custom_info_buffer = deque(maxlen=self._stats_window_size)
        temp_dict["torque"] = infos["tor"]
        temp_dict["to_reach"] = infos["dst"]
        temp_dict["from_init"] = infos["init"]
        self._custom_info_buffer.extend([temp_dict])


class SaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output,
    1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        config_filename: Optional[str] = None,
        training_name: Optional[str] = None,
        check_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.save_path = None
        self.config_filename = config_filename
        self.training_name = training_name

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.save_path is None:
            self.save_path = self.locals["self"].logger.dir
            shutil.copy(
                self.config_filename,
                self.save_path + "/" + self.training_name + ".yaml",
            )

    def _on_step(self) -> bool:
        try:
            self._update_info_buffer_save(self.locals["infos"][0]["episode"])
        except:  # noqa: E722
            pass
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            if len(self._ep_info_buffer) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = safe_mean(list(self._ep_info_buffer))
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Best model found with mean of: {mean_reward:.2f}")
                        print(
                            f"Saving new best model to {self.save_path}/best_model.zip",
                        )
                    self.model.save(self.save_path + "/" + "best_model.zip")
        return True


class AllCallbacks(BaseCallback):
    def __init__(
        self,
        config_filename: Optional[str] = None,
        training_name: Optional[str] = None,
        stats_window_size: int = 100,
        check_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.config_filename = config_filename
        self.training_name = training_name
        self.best_mean_reward = -np.inf
        self._stats_window_size = stats_window_size
        self._custom_info_buffer = None
        self._episode_num = 0
        self._ep_info_buffer = None
        self.save_path = None
        self._ep_end_buffer = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.save_path = self.locals["self"].logger.dir
        shutil.copy(
            self.config_filename,
            self.save_path + "/" + self.training_name + ".yaml",
        )

    def _on_step_save(self) -> bool:
        try:
            self._update_info_buffer_save(self.locals["infos"][0]["episode"])
        except:  # noqa: E722
            pass
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            if self._ep_info_buffer is not None:
                if len(self._ep_info_buffer) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = safe_mean(
                        list(self._ep_info_buffer),
                    )
                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward and mean_reward > 0:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose >= 1:
                            print(f"Best model found with mean of: {mean_reward:.2f}")
                            print(
                                f"New best model at {self.save_path}/best_model.zip",
                            )
                        self.model.save(self.save_path + "/" + "best_model.zip")

    def _on_step_tensor(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._update_info_buffer_tensor(self.locals["infos"][0])
        if self.locals["dones"][0]:
            self._episode_num += 1
            self._ep_end_buffer.extend([self.locals["infos"][0]["dst"]])
            if (
                self.locals["log_interval"] is not None
                and self._episode_num % self.locals["log_interval"] == 0
            ):
                self._dump_logs_tensor()

    def _dump_logs_tensor(self) -> None:
        """
        Write log.
        """
        if len(self._custom_info_buffer) > 0:
            self.logger.record(
                "z_custom/torque_mean",
                safe_mean([ep_info["torque"] for ep_info in self._custom_info_buffer]),
            )
            self.logger.record(
                "z_custom/to_reach_mean",
                safe_mean(
                    [ep_info["to_reach"] for ep_info in self._custom_info_buffer],
                ),
            )
            self.logger.record(
                "z_custom/from_init_mean",
                safe_mean(
                    [ep_info["from_init"] for ep_info in self._custom_info_buffer],
                ),
            )
            self.logger.record(
                "z_custom/final_dt",
                safe_mean(list(self._ep_end_buffer)),
            )

        # self.logger.dump(step=self.num_timesteps)
        return True

    def _update_info_buffer_tensor(self, infos):
        """
        Update the buffer for episode infos.
        :param infos: ([dict]) List of infos
        """
        temp_dict = {}
        if self._custom_info_buffer is None:
            self._custom_info_buffer = deque(maxlen=self._stats_window_size)
        if self._ep_end_buffer is None:
            self._ep_end_buffer = deque(maxlen=self._stats_window_size // 2)
        temp_dict["torque"] = infos["tor"]
        temp_dict["to_reach"] = infos["dst"]
        temp_dict["from_init"] = infos["init"]
        self._custom_info_buffer.extend([temp_dict])

    def _update_info_buffer_save(self, infos):
        """
        Update the buffer for episode infos.
        :param infos: ([dict]) List of infos
        """
        if self._ep_info_buffer is None:
            self._ep_info_buffer = deque(maxlen=self._stats_window_size)
        self._ep_info_buffer.extend([infos["r"]])

    def _on_step(self) -> bool:
        self._on_step_tensor()
        self._on_step_save()
        return True
