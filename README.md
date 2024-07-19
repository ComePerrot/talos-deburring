# talos-manipulation

Set of tools to work on manipulation tasks with the robot Talos.

This repo is built around several packages:

- [deburring_deep_planner](./deburring-deep-planner/README.md): a rosnode to do the inferance of a neural-network
- [deburring_mpc](./deburring-mpc/README.md): an implementation of a Model Predictive Controller based on Crocoddyl
- [deburring_python_utils](./deburring-python-utils/README.md): a collection of usefull python tools
- [deburring_ros_interface](./deburring-ros-interface/README.md): the interface to allow the use of the MPC within a ROS architecture

As well as some python examples (which can be found in pyTalos)

## Copyrights and License

This package is distributed under a [BSD-2-Clause Licence](./LICENSE).

This repo was initially a fork of [sobec](https://github.com/MeMory-of-MOtion/sobec).

Authored by:

- Côme Perrot