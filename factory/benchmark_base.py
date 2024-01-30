import pinocchio as pin
import numpy as np
import yaml

from limit_checker.limit_checker import LimitChecker


class bench_base:
    def __init__(self, filename, pinWrapper, simulator):
        # PARAMETERS
        with open(filename, "r") as paramFile:
            self.params = yaml.safe_load(paramFile)

        self.error_tolerance = self.params["toleranceError"]
        self.time_step_simulation = float(self.params["timeStepSimulation"])
        self.maxTime = int(self.params["maxTime"] / self.time_step_simulation)

        # Target
        self.oMtarget = pin.SE3.Identity()
        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        # Robot handler
        self.pinWrapper = pinWrapper
        self.limit_checker = LimitChecker(self.pinWrapper.get_rmodel(), False)

        # Simulator
        self.simulator = simulator

        self._define_controller()

    def reset(self, target_position):
        for i in range(3):
            self.oMtarget.translation[i] = target_position[i]
        # Reset simulator
        self.simulator.reset(target_pos=target_position)

        self._reset_controller()

    def run(self, target_position):
        self.reset(target_position)

        # Initialization
        #   Reset Variables
        Time = 0
        target_reached = False
        reach_time = None

        # Control loop
        while Time < self.maxTime:
            x_measured = self.simulator.getRobotState()

            torques = self._run_controller(Time, x_measured)

            self.simulator.step(torques, self.pinWrapper.get_end_effector_frame())

            Time += 1

            self.pinWrapper.update_reduced_model(self.simulator.getRobotState())

            error_placement_tool = np.linalg.norm(
                self.pinWrapper.get_end_effector_frame().translation
                - self.oMtarget.translation
            )

            if error_placement_tool < self.error_tolerance:
                if not target_reached:
                    target_reached = True
                    reach_time = Time * self.time_step_simulation
            else:
                target_reached = False
                reach_time = None

            limits = self.limit_checker.are_limits_broken(
                x_measured[7 : self.pinWrapper.get_rmodel().nq],
                x_measured[-self.pinWrapper.get_rmodel().nv + 6 :],
                torques,
            )
            if limits is not False:
                break

        return (
            reach_time,
            error_placement_tool,
            limits
        )

    def _check_limits(self, x, torques):
        exceeded_position_list = []
        exceeded_speed_list = []
        exceeded_torque_list = []

        rmodel = self.pinWrapper.get_rmodel()

        for jointName in rmodel.names[2:]:
            id = rmodel.getJointId(jointName) - 2

            position = x[7 + id]
            speed = x[rmodel.nq + 6 + id]
            torque = torques[id]

            if (position > rmodel.upperPositionLimit[7 + id]) or (
                position < rmodel.lowerPositionLimit[7 + id]
            ):
                if position > rmodel.upperPositionLimit[7 + id]:
                    exceeded_position_list.append(
                        (jointName, position, rmodel.upperPositionLimit[7 + id])
                    )
                else:
                    exceeded_position_list.append(
                        (jointName, position, rmodel.lowerPositionLimit[7 + id])
                    )
            if np.abs(speed) > rmodel.velocityLimit[6 + id]:
                exceeded_speed_list.append(
                    (jointName, speed, rmodel.velocityLimit[6 + id])
                )
            if np.abs(torque) > rmodel.effortLimit[6 + id]:
                exceeded_torque_list.append(
                    (jointName, torque, rmodel.effortLimit[6 + id])
                )
        return (exceeded_position_list, exceeded_speed_list, exceeded_torque_list)

    def _define_controller(self):
        raise NotImplementedError()

    def _reset_controller(self):
        raise NotImplementedError()

    def _run_controller(self, Time, x_measured):
        raise NotImplementedError()
