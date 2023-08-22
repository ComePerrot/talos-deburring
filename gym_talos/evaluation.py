from pathlib import Path

import yaml
from stable_baselines3 import SAC

from .envs.env_talos_deburring_her import EnvTalosDeburringHer
from .envs.env_talos_mpc_deburring import EnvTalosMPC
from .utils.create_target import TargetGoal

# Script parameters
envMPC = True
bestModel = True
training_name = "2023-08-18_one_joint_several_targets_2"

threshold = 0.005

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
        GUI=False,
    )
else:
    envDisplay = EnvTalosDeburringHer(
        params["robot_designer"],
        params["environment"],
        GUI=False,
    )

model = SAC.load(model_path, env=envDisplay)

record_success = []
record_final_dst = []
record_time = []

target_builder = TargetGoal(params["environment"])
targets = target_builder.generate_target(100)
for target in targets:
    obs, infos = envDisplay.reset(options={"target": target})
    done = False
    i = 0
    while not done:
        i += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, infos = envDisplay.step(action)
        if truncated:
            record_success.append(0)
            done = True
        elif terminated:
            time = infos["time"]
            distance = infos["dst"]

            if distance < threshold:
                record_success.append(1)
                print(float(time)*0.2)
            else:
                record_success.append(0)
                print(time)

            done = True

print(sum(record_success))
