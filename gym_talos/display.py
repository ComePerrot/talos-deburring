from pathlib import Path

import yaml
from stable_baselines3 import SAC

from .envs.env_talos_deburring import EnvTalosDeburring
from .envs.env_talos_deburring_her import EnvTalosDeburringHer

training_name = "2023-06-23_var_8"

log_dir = Path("logs")
model_path = log_dir / training_name / f"{training_name[:-2]}.zip"
config_path = log_dir / training_name / f"{training_name[:-2]}.yaml"
with config_path.open() as config_file:
    params = yaml.safe_load(config_file)

envDisplay = EnvTalosDeburringHer(
    params["robot_designer"],
    params["environment"],
    GUI=True,
)

model = SAC.load(model_path, env=envDisplay)


envDisplay.maxTime = 10000
obs, info = envDisplay.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = envDisplay.step(action)
    print("Reward: ", reward)
    print("Obs: ", obs)
    if terminated or truncated:
        obs, info = envDisplay.reset()