# talos-manipulation

Set of tools to work on manipulation tasks with the robot Talos.

This repo is built around several packages:

- [deburring_deep_planner](./deburring-deep-planner/README.md): a rosnode to do the inference of a neural-network
- [deburring_mpc](./deburring-mpc/README.md): an implementation of a Model Predictive Controller based on Crocoddyl
- [deburring_python_utils](./deburring-python-utils/README.md): a collection of useful python tools
- [deburring_ros_interface](./deburring-ros-interface/README.md): the interface to allow the use of the MPC within a ROS architecture

## Usage

The Apptainer is designed to provide a solution for running trainings of the RL policy or benchmarks.
It is not intended for use when conducting experiments on the robot, thus it does not contain the `deburring_ros_interface`.

Build apptainer:

```bash
apptainer build rl.sif apptainer/rl_mamba.def
```

Several apps are available:

- default training:

Short training with default parameters.

```bash
apptainer run --app default_training_mpc rl.sif
```

- training:

Training with custom parameters.
This training requires a configuration file to be mount in the apptainer:

```bash
apptainer run --app training_mpc \
--bind path/to/config:/config \
rl.sif
```

An example of the config file to provide can be found in `deburring_python_utils/gym_talos/config/config_MPC_RL.yaml`.

- benchmark:

```bash
apptainer run --app benchmark rl.sif
```

The benchmark is carried out the configuration and the example policy that can be found in `deburring_python_utils/deburring_benchmark/`.

## Copyrights and License

This package is distributed under a [BSD-2-Clause Licence](./LICENSE).

This repo was initially a fork of [sobec](https://github.com/MeMory-of-MOtion/sobec).

Authored by:

- Côme Perrot
