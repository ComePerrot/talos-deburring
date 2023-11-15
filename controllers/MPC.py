import numpy as np
import pinocchio as pin

from deburring_mpc import OCP


class MPController:
    def __init__(self, pinWrapper, x_initial, target_pos, param_ocp):
        self.num_control_knots = 10

        self.pinWrapper = pinWrapper
        self.param_ocp = param_ocp

        self.param_ocp["state_weights"] = np.array(self.param_ocp["state_weights"])
        self.param_ocp["control_weights"] = np.array(self.param_ocp["control_weights"])

        
        self.oMtarget = pin.SE3.Identity()
        for i in range(3):
            self.oMtarget.translation[i] = target_pos[i]

        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.crocoWrapper = OCP(self.param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(x_initial, self.oMtarget)

        self.horizon_length = param_ocp["horizon_length"]

        self.x0 = x_initial

    def change_target(self, x_initial, target_position):
        for i in range(3):
            self.oMtarget.translation[i] = target_position[i]

        self.crocoWrapper = OCP(self.param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(x_initial, self.oMtarget)

    def step(self, x_measured, reference_posture=None):
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

        return (self.crocoWrapper.torque, self.x0, self.crocoWrapper.gain)
