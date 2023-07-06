import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self,
                    log_dir: str = None, 
                    stats_window_size: int = 100,
                    check_freq: int = 1000,
                    verbose: int = 0):
        super().__init__(verbose)
        self._stats_window_size = stats_window_size
        self._custom_info_buffer = None
        self._episode_num = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._update_info_buffer(self.locals['infos'][0])
        if self.locals['dones'][0]:
            self._episode_num += 1
            if self.locals['log_interval'] is not None and self._episode_num % self.locals['log_interval'] == 0:
                self._dump_logs()

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        if len(self._custom_info_buffer) > 0:
            self.logger.record("z_custom/torque_mean", safe_mean([ep_info["torque"] for ep_info in self._custom_info_buffer]))
            self.logger.record("z_custom/to_reach_mean", safe_mean([ep_info["to_reach"] for ep_info in self._custom_info_buffer]))
            self.logger.record("z_custom/from_init_mean", safe_mean([ep_info["from_init"] for ep_info in self._custom_info_buffer]))


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



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,
                    log_dir: str = None, 
                    stats_window_size: int = 100,
                    check_freq: int = 1000,
                    verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class AllCallbacks(BaseCallback):
    def __init__(self,
                    log_dir: str = None, 
                    stats_window_size: int = 100,
                    check_freq: int = 1000,
                    verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self._stats_window_size = stats_window_size
        self._custom_info_buffer = None
        self._episode_num = 0
        self._ep_info_buffer = None

    def _init_callback_save(self) -> None:
        # Create folder if needed
        self.model.save(self.save_path)

    def _on_step_save(self) -> bool:
        try:
            self._update_info_buffer_save(self.locals['infos'][0]['episode'])
        except:
            pass
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          if len(self._ep_info_buffer) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = safe_mean([ep_info for ep_info in self._ep_info_buffer])
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Best model found with mean of: {mean_reward:.2f}")
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
    
    def _on_step_tensor(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._update_info_buffer_tensor(self.locals['infos'][0])
        if self.locals['dones'][0]:
            self._episode_num += 1
            if self.locals['log_interval'] is not None and self._episode_num % self.locals['log_interval'] == 0:
                self._dump_logs_tensor()

    def _dump_logs_tensor(self) -> None:
        """
        Write log.
        """
        if len(self._custom_info_buffer) > 0:
            self.logger.record("z_custom/torque_mean", safe_mean([ep_info["torque"] for ep_info in self._custom_info_buffer]))
            self.logger.record("z_custom/to_reach_mean", safe_mean([ep_info["to_reach"] for ep_info in self._custom_info_buffer]))
            self.logger.record("z_custom/from_init_mean", safe_mean([ep_info["from_init"] for ep_info in self._custom_info_buffer]))


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
        self._ep_info_buffer.extend([infos['r']])

    def _init_callback(self) -> None:
        self._init_callback_save()
        return True
    
    def _on_step(self) -> bool:
        self._on_step_tensor()
        self._on_step_save()
        return True