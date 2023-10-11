import numpy as np
import pinocchio as pin

from deburring_mpc import OCP, RobotDesigner


class mpc_wrapper:
    def __init__(self, x_initial, target_pos, params_designer, param_ocp):
        self.time = 0
        self.num_control_knots = 10

        self.pinWrapper = RobotDesigner()
        self.pinWrapper.initialize(params_designer)

        gripper_SE3_tool = pin.SE3.Identity()
        gripper_SE3_tool.translation[0] = params_designer["toolPosition"][0]
        gripper_SE3_tool.translation[1] = params_designer["toolPosition"][1]
        gripper_SE3_tool.translation[2] = params_designer["toolPosition"][2]
        self.pinWrapper.add_end_effector_frame(
            "deburring_tool",
            "gripper_left_fingertip_3_link",
            gripper_SE3_tool,
        )

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

    def step(self, x_measured):
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

            self.crocoWrapper.solve(x_measured)

        return torques
