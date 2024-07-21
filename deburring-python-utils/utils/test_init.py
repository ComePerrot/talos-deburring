from pathlib import Path

import numpy as np
import yaml
from deburring_mpc import RobotDesigner

from controllers.MPC import MPController
from gym_talos.utils.create_target import TargetGoal


def check_node_limits(pinWrapper, x, torques):
    # Limits
    limit_position = (
        x[: pinWrapper.get_rmodel().nq] > pinWrapper.get_rmodel().upperPositionLimit
    ).any() or (
        x[: pinWrapper.get_rmodel().nq] < pinWrapper.get_rmodel().lowerPositionLimit
    ).any()
    limit_speed = (
        np.abs(x[-pinWrapper.get_rmodel().nv :]) > pinWrapper.get_rmodel().velocityLimit
    ).any()
    limit_command = (np.abs(torques) > pinWrapper.get_rmodel().effortLimit[6:]).any()
    return (limit_position, limit_speed, limit_command)


def check_horizon_limits(solver, pinWrapper):
    us_list = solver.us.tolist()
    xs_list = solver.xs.tolist()

    for i in range(len(us_list)):
        xs = xs_list[i]
        us = us_list[i]
        limit_position, limit_speed, limit_command = check_node_limits(
            pinWrapper,
            xs,
            us,
        )
        if limit_position:
            return (0, i)
        if limit_position:
            return (1, i)
        if limit_command:
            return (2, i)
    else:
        return None


def main():
    # PARAMETERS
    filename = "config/config.yaml"
    with Path.open(filename, "r") as paramFile:
        params = yaml.safe_load(paramFile)

    target_handler = TargetGoal(params["target"])
    target_handler.create_target()
    targets = target_handler.generate_target_list(params["numberTargets"])

    # Robot handler
    pinWrapper = RobotDesigner()
    params["robot"]["end_effector_position"] = np.array(
        params["robot"]["end_effector_position"],
    )
    pinWrapper.initialize(params["robot"])

    # OCP
    MPC = MPController(
        pinWrapper,
        pinWrapper.get_x0(),
        targets[0],
        params["OCP"],
    )

    for target in targets:
        print("Testing for target: " + str(np.array(target)))
        MPC.change_target(pinWrapper.get_x0(), target)

        for i in range(200):
            limits_exceeded = check_horizon_limits(MPC.crocoWrapper.solver, pinWrapper)
            if limits_exceeded is not None:
                limit_type, node = limits_exceeded
                print("Limits broken:")
                if limit_type == 0:
                    print(" - position")
                elif limit_type == 1:
                    print(" - speed")
                elif limit_type == 2:
                    print(" - command")
                print(" - node " + str(node))
                print(" - iteration " + str(i))
            MPC.step(MPC.crocoWrapper.solver.xs[1])
        else:
            print("Movement successfully carried out without breaking the limits")


if __name__ == "__main__":
    main()
