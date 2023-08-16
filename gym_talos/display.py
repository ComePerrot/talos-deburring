from pathlib import Path

import yaml
from stable_baselines3 import SAC

from .envs.env_talos_deburring_her import EnvTalosDeburringHer

# The name of the file who needs to be displayed
training_name = "2023-08-16_local_2_10_1e-1_4_1"
train_name = "_".join(training_name.split("_")[:-1])

# The path to the file
log_dir = Path("logs")

# Depends on the version of the files, sometimes best_model, sometimes complete model
# model_path = log_dir / training_name / f"{train_name}.zip"
model_path = log_dir / training_name / "best_model.zip"
config_path = log_dir / training_name / f"{train_name}.yaml"
with config_path.open() as config_file:
    params = yaml.safe_load(config_file)

envDisplay = EnvTalosDeburringHer(
    params["robot_designer"],
    params["environment"],
    GUI=True,
)

model = SAC.load(model_path, env=envDisplay)

while True:
    obs, info = envDisplay.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, infos = envDisplay.step(action)
        if terminated or truncated:
            done = True
            print("Time", i)
            print("Infos", infos)
