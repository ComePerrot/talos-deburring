from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from controllers.RL_posture import RLPostureController

from deburring_benchmark.factory.benchmark_base import bench_base


class bench_MPRL(bench_base):
    def __init__(self, filename, model_path, target_handler, pinWrapper, simulator):
        self.target_handler = target_handler
        self.model_path = model_path
        super().__init__(filename, pinWrapper, simulator)

    def _define_controller(self):
        #   RL Posture controller
        #       Action wrapper
        controlled_joints_names = self.params["RL_posture"]["controlled_joints_names"]
        kwargs_action = {
            "rmodel": self.pinWrapper.get_rmodel(),
            "rl_controlled_joints": controlled_joints_names,
            "initial_state": self.pinWrapper.get_x0(),
            "scaling_factor": self.params["RL_posture"]["actionScale"],
            "scaling_mode": self.params["RL_posture"]["actionType"],
        }
        #       Observation wrapper
        kwargs_observation = {
            "normalize_obs": self.params["RL_posture"]["normalizeObs"],
            "rmodel": self.pinWrapper.get_rmodel(),
            "target_handler": self.target_handler,
            "history_size":  self.params["RL_posture"]["historyObs"],
            "prediction_size": self.params["RL_posture"]["predictionSize"],
        }
        self.posture_controller = RLPostureController(
            self.model_path,
            self.pinWrapper.get_x0().copy(),
            kwargs_action,
            kwargs_observation,
        )

        # MPC
        self.mpc = MPController(
            self.pinWrapper,
            self.pinWrapper.get_x0(),
            self.oMtarget.translation,
            self.params["OCP"],
            self.params["MPC_delay"],
        )

        # RICCATI
        self.riccati = RiccatiController(
            state=self.mpc.crocoWrapper.state,
            torque=self.mpc.crocoWrapper.torque,
            xref=self.pinWrapper.get_x0(),
            riccati=self.mpc.crocoWrapper.gain,
        )

        time_step_OCP = float(self.params["OCP"]["time_step"])
        self.num_simulation_step = int(time_step_OCP / self.time_step_simulation)
        self.num_OCP_steps = int(self.params["RL_posture"]["numOCPSteps"])

    def _reset_controller(self):
        self.mpc.change_target(self.pinWrapper.get_x0(), self.oMtarget.translation)
        self.posture_controller.observation_wrapper.reset(
            self.pinWrapper.get_x0(),
            self.oMtarget.translation,
            self.mpc.crocoWrapper.solver.xs,
        )

    def _run_controller(self, Time, x_measured):
        if Time % (self.num_simulation_step * self.num_OCP_steps) == 0:
            self.x_reference = self.posture_controller.step(
                x_measured,
                self.mpc.crocoWrapper.solver.xs,
            )
            if self.simulator.enable_GUI == 2:
                self.simulator.posture_visualizer.update_posture(
                    self.x_reference[7 : self.pinWrapper.get_rmodel().nq],
                )

        if Time % self.num_simulation_step == 0:
            t0, x0, K0 = self.mpc.step(x_measured, self.x_reference)
            self.riccati.update_references(t0, x0, K0)

        torques = self.riccati.step(x_measured)

        return torques  # noqa: RET504
