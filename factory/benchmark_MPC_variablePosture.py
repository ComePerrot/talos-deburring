import numpy as np

from factory.benchmark_MPC import bench_MPC


class bench_MPC_variablePosture(bench_MPC):
    def _run_controller(self, Time, x_measured):
        if Time % self.num_simulation_step == 0:
            x_reference = x_measured
            x_reference[self.pinWrapper.get_rmodel().nq:] = np.zeros(
                self.pinWrapper.get_rmodel().nv
            )

            t0, x0, K0 = self.mpc.step(x_measured, x_reference)
            self.riccati.update_references(t0, x0, K0)

        torques = self.riccati.step(x_measured)

        return torques  # noqa: RET504
