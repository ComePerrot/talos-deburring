import pinocchio as pin
import numpy as np
import yaml


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

            self.simulator.step(
                torques, self.pinWrapper.get_end_effector_frame(), self.oMtarget
            )

            Time += 1

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

        return (reach_time, error_placement_tool)

    def _define_controller(self):
        raise NotImplementedError()

    def _reset_controller(self):
        raise NotImplementedError()

    def _run_controller(self, Time, x_measured):
        raise NotImplementedError()
