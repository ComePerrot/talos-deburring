from factory.benchmark_base import bench_base
from controllers.MPC import MPController
from controllers.Riccati import RiccatiController


class bench_MPC(bench_base):
    def _define_controller(self):
        # MPC
        self.mpc = MPController(
            self.pinWrapper,
            self.pinWrapper.get_x0(),
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

        time_step_OCP = float(self.params["OCP"]["time_step"])
        self.num_simulation_step = int(time_step_OCP / self.time_step_simulation)

    def _reset_controller(self):
        # Reset OCP
        self.mpc.change_target(self.pinWrapper.get_x0(), self.oMtarget.translation)

    def _run_controller(self, Time, x_measured):
        if Time % self.num_simulation_step == 0:
            t0, x0, K0 = self.mpc.step(x_measured, None)
            self.riccati.update_references(t0, x0, K0)

        torques = self.riccati.step(x_measured)

        return torques
