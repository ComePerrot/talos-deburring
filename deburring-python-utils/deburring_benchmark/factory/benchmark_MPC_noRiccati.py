from controllers.MPC import MPController
from deburring_benchmark.factory.benchmark_base import bench_base


class bench_MPC_noRiccati(bench_base):
    def _define_controller(self):
        # MPC
        self.mpc = MPController(
            self.pinWrapper,
            self.pinWrapper.get_x0(),
            self.oMtarget.translation,
            self.params["OCP"],
            self.params["MPC_delay"],
        )

        time_step_OCP = float(self.params["OCP"]["time_step"])
        self.num_simulation_step = int(time_step_OCP / self.time_step_simulation)

        self.x_list = []
        self.u_list = []

    def _reset_controller(self):
        # Reset OCP
        self.mpc.change_target(self.pinWrapper.get_x0(), self.oMtarget.translation)

    def _run_controller(self, Time, x_measured):
        if Time % self.num_simulation_step == 0:
            self.torques, _, _ = self.mpc.step(x_measured, None)

        self.x_list.append(x_measured)
        self.u_list.append(self.torques)

        return self.torques
