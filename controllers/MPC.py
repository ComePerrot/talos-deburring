import numpy as np
import pinocchio as pin

from deburring_mpc import OCP


class MPController:
    def __init__(self, pinWrapper, x_initial, target_pos, param_ocp):
        self.time = 0
        self.num_control_knots = 10

        self.pinWrapper = pinWrapper

        self.param_ocp["state_weights"] = np.array(self.param_ocp["state_weights"])
        self.param_ocp["control_weights"] = np.array(self.param_ocp["control_weights"])

        self.oMtarget = pin.SE3.Identity()
        self.oMtarget.translation[0] = target_pos[0]
        self.oMtarget.translation[1] = target_pos[1]
        self.oMtarget.translation[2] = target_pos[2]

        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.crocoWrapper = OCP(param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(x_initial, self.oMtarget)

        self.x0 = x_initial

    def step(self, x_measured, reference_posture=None):
        self.time += 1

        # Compute torque to be applied by adding Riccati term
        torques = (
            self.crocoWrapper.torque
            + self.crocoWrapper.gain @ self.crocoWrapper.state.diff(x_measured, self.x0)
        )

        if self.time % self.num_control_knots == 0:
            self.x0 = x_measured

            self.pinWrapper.update_reduced_model(x_measured)

            self.crocoWrapper.recede()
            self.crocoWrapper.change_goal_cost_activation(self.horizon_length - 1, True)

            if reference_posture is not None:
                self.crocoWrapper.change_posture_reference(
                    self.horizon_length - 1,
                    reference_posture,
                )
                self.crocoWrapper.change_posture_reference(
                    self.horizon_length,
                    reference_posture,
                )

            self.crocoWrapper.solve(x_measured)

        return torques
