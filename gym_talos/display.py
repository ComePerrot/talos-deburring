from pathlib import Path

import yaml
from stable_baselines3 import SAC

from .envs.env_talos_deburring_her import EnvTalosDeburringHer
from .envs.env_talos_mpc_deburring import EnvTalosMPC

# Script parameters
envMPC = True
bestModel = True
training_name = "2023-09-04_severalJoints_severalTargets_predictions_1"


train_name = "_".join(training_name.split("_")[:-1])
log_dir = Path("logs")
if bestModel:
    model_path = log_dir / training_name / "best_model.zip"
else:
    model_path = log_dir / training_name / f"{train_name}.zip"
config_path = log_dir / training_name / f"{train_name}.yaml"
with config_path.open() as config_file:
    params = yaml.safe_load(config_file)

if envMPC:
    envDisplay = EnvTalosMPC(
        params["robot"],
        params["environment"],
        GUI=True,
    )
    print(params["environment"]["controlled_joints_names"])
    envDisplay.simulator.setupPostureVisualizer(
        params["environment"]["controlled_joints_names"],
    )
else:
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
        envDisplay.simulator.updatePosture(action)
        obs, reward, terminated, truncated, infos = envDisplay.step(action)
        if terminated or truncated:
            done = True
            print("Time", i)
            print("Infos", infos)
