import numpy as np

from factory.benchmark_base import bench_base
from controllers.MPC import MPController
from controllers.Riccati import RiccatiController
from controllers.RL_posture import RLPostureController


class bench_MPRL(bench_base):
    def __init__(self, filename, target_handler, pinWrapper, simulator):
        self.target_handler = target_handler
        super().__init__(filename, pinWrapper, simulator)

    def _define_controller(self):
        #   RL Posture controller
        #       Action wrapper
        controlled_joints_names = self.params["RL_posture"]["controlled_joints_names"]
        rl_controlled_IDs = np.array(
            [
                self.pinWrapper.get_rmodel().names.tolist().index(joint_name) - 2 + 7
                for joint_name in controlled_joints_names
            ],
        )
        kwargs_action = dict(
            rl_controlled_IDs=rl_controlled_IDs,
            rmodel=self.pinWrapper.get_rmodel(),
            scaling_factor=self.params["RL_posture"]["actionScale"],
            scaling_mode=self.params["RL_posture"]["actionType"],
            initial_pose=None,
        )
        #       Observation wrapper
        kwargs_observation = dict(
            normalize_obs=True,
            rmodel=self.pinWrapper.get_rmodel(),
            target_handler=self.target_handler,
            history_size=0,
            prediction_size=3,
        )
        model_path = self.params["RL_posture"]["model_path"]
        self.posture_controller = RLPostureController(
            model_path,
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
                x_measured, self.mpc.crocoWrapper.solver.xs
            )

        if Time % self.num_simulation_step == 0:
            t0, x0, K0 = self.mpc.step(x_measured, self.x_reference)
            self.riccati.update_references(t0, x0, K0)

        torques = self.riccati.step(x_measured)

        return torques
