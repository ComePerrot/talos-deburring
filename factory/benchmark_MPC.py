import pinocchio as pin
import numpy as np
import yaml

from controllers.MPC import MPController
from controllers.Riccati import RiccatiController


class bench_MPC:
    def __init__(self, filename, target_handler, pinWrapper, simulator):
        # PARAMETERS
        with open(filename, "r") as paramFile:
            self.params = yaml.safe_load(paramFile)

        # Target
        self.oMtarget = pin.SE3.Identity()
        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        # Robot handler
        self.pinWrapper = pinWrapper

        # Simulator
        self.simulator = simulator

        # Controllers
        # MPC
        self.mpc = MPController(
            pinWrapper,
            pinWrapper.get_x0(),
            self.oMtarget.translation,
            self.params["OCP"],
        )

        # RICCATI
        self.riccati = RiccatiController(
            state=self.mpc.crocoWrapper.state,
            torque=self.mpc.crocoWrapper.torque,
            xref=self.pinWrapper.get_x0(),
            riccati=self.mpc.crocoWrapper.gain,
        )

        # Parameters
        self.error_tolerance = self.params["toleranceError"]
        #   Timings
        time_step_simulation = float(self.params["timeStepSimulation"])
        time_step_OCP = float(self.params["OCP"]["time_step"])
        self.maxTime = int(self.params["maxTime"] / time_step_simulation)
        self.num_simulation_step = int(time_step_OCP / time_step_simulation)
        self.num_OCP_steps = int(self.params["RL_posture"]["numOCPSteps"])

    def reset(self, target_position):
        for i in range(3):
            self.oMtarget.translation[i] = target_position[i]
        # Reset simulator
        self.simulator.reset(target_pos=target_position)
        # Reset OCP
        self.mpc.change_target(self.pinWrapper.get_x0(), target_position)

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

            if Time % self.num_simulation_step == 0:
                t0, x0, K0 = self.mpc.step(x_measured, None)
                self.riccati.update_references(t0, x0, K0)

            torques = self.riccati.step(x_measured)
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
                    reach_time = Time
                    print(reach_time)
            else:
                target_reached = False
