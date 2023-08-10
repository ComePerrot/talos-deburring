from pathlib import Path

import yaml
from stable_baselines3 import SAC

from .envs.env_talos_deburring_her import EnvTalosDeburringHer
from .utils.create_target import TargetGoal

training_name = "2023-08-09_local_2_10_1e-1_4_8"
train_name = "_".join(training_name.split("_")[:-1])


log_dir = Path("logs")
model_path = log_dir / training_name / "best_model.zip"
config_path = log_dir / training_name / f"{train_name}.yaml"
with config_path.open() as config_file:
    params = yaml.safe_load(config_file)

envDisplay = EnvTalosDeburringHer(
    params["robot_designer"],
    params["environment"],
    GUI=False,
)

model = SAC.load(model_path, env=envDisplay)

record_success = []
record_final_dst = []
record_time = []

target_builder = TargetGoal(envDisplay.params_env)
targets = target_builder.generate_target(100)
for target in targets:
    obs, info = envDisplay.reset(options={"target": target})
    done = False
    i = 0
    while not done:
        i += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, infos = envDisplay.step(action)
        if terminated or truncated:
            done = True
            record_success.append(1) if infos["is_success"] else record_success.append(
                0,
            )
            record_final_dst.append(infos["dst"])
            record_time.append(i)
            print("Episode done with:")
            print("Success", infos["is_success"])
            print("Final distance", infos["dst"])
            print("Time", i)
