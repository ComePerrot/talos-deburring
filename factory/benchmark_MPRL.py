import pinocchio as pin
import numpy as np
import yaml

from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from controllers.RL_posture import RLPostureController


class bench_MPRL:
    def __init__(self, filename, pinWrapper, simulator, target_handler):
        # PARAMETERS
        with open(filename, "r") as paramFile:
            self.params = yaml.safe_load(paramFile)

        oMtarget = pin.SE3.Identity()
        oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        # Robot handler
        self.pinWrapper = pinWrapper

        # SIMULATOR
        self.simulator = simulator

        # CONTROLLERS
        #   RL Posture controller
        #       Action wrapper
        kwargs_action = dict(
            rl_controlled_IDs=[16, 17, 19],
            rmodel=pinWrapper.get_rmodel(),
            scaling_factor=self.params["RL_posture"]["actionScale"],
            scaling_mode=self.params["RL_posture"]["actionType"],
            initial_pose=None,
        )
        #       Observation wrapper
        kwargs_observation = dict(
            normalize_obs=True,
            rmodel=pinWrapper.get_rmodel(),
            target_handler=target_handler,
            history_size=0,
            prediction_size=3,
        )
        model_path = "config/2023-11-08_3joints_fullRange_4/best_model.zip"
        self.posture_controller = RLPostureController(
            model_path, pinWrapper.get_x0().copy(), kwargs_action, kwargs_observation
        )

        # MPC
        self.mpc = MPController(
            pinWrapper,
            pinWrapper.get_x0(),
            pin.SE3.Identity().translation(),
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
        self.maxTime = int(
            self.params["maxTime"] / float(self.params["timeStepSimulation"])
        )
        self.num_simulation_step = int(
            float(self.params["OCP"]["time_step"])
            / float(self.params["timeStepSimulation"])
        )
        self.num_OCP_steps = int(self.params["RL_posture"]["numOCPSteps"])

    def reset():
        pass

    def run(self, target_position):
        self.target_handler.set_target(target_position)
        # Initialization
        #   Reset Variables
        Time = 0
        target_reached = False
        reach_time = None
        xfuture_list = self.mpc.crocoWrapper.solver.xs
        self.posture_controller.observation_wrapper.reset(
            self.pinWrapper.get_x0(), target_position, xfuture_list
        )

        # Control loop
        while Time < self.maxTime:
            x_measured = self.simulator.getRobotState()

            if Time % (self.num_simulation_step * self.num_OCP_steps) == 0:
                x_reference = self.posture_controller.step(x_measured, xfuture_list)

            if Time % self.num_simulation_step == 0:
                t0, x0, K0 = self.mpc.step(x_measured, x_reference)
                self.riccati.update_references(t0, x0, K0)
                xfuture_list = self.mpc.crocoWrapper.solver.xs

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
            else:
                target_reached = False
