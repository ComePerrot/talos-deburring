import numpy as np
import pinocchio as pin
import queue

from deburring_mpc import OCP


class MPController:
    def __init__(self, pinWrapper, x_initial, target_pos, param_ocp, delay=0):
        self.output_queue = queue.Queue()
        self.delay = delay

        self.pinWrapper = pinWrapper
        self.param_ocp = param_ocp

        self._get_joint_parameters()

        self.oMtarget = pin.SE3.Identity()
        for i in range(3):
            self.oMtarget.translation[i] = target_pos[i]

        self.oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.crocoWrapper = OCP(self.param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(x_initial, self.oMtarget)

        self.horizon_length = param_ocp["horizon_length"]

        self.x0 = x_initial

    def _get_joint_parameters(self):
        # Retrieve controlled joint names
        controlled_joints_names = self.pinWrapper.get_rmodel().names[1:]

        # Check if robot has a free-flyer
        free_flyer = "root_joint" in controlled_joints_names

        joint_position_weights = [
            self.param_ocp["joints"][joint][0]
            for joint in controlled_joints_names[1 * free_flyer :]
        ]
        joint_velocity_weights = [
            self.param_ocp["joints"][joint][1]
            for joint in controlled_joints_names[1 * free_flyer :]
        ]

        if free_flyer:
            self.param_ocp["state_weights"] = np.concatenate(
                [
                    self.param_ocp["base"]["position"],
                    self.param_ocp["base"]["orientation"],
                    joint_position_weights,
                    self.param_ocp["base"]["linear_velocity"],
                    self.param_ocp["base"]["angular_velocity"],
                    joint_velocity_weights,
                ],
            )
        else:
            self.param_ocp["state_weights"] = np.concatenate(
                [
                    joint_position_weights,
                    joint_velocity_weights,
                ],
            )

        self.param_ocp["control_weights"] = np.array(
            [
                self.param_ocp["joints"][joint][2]
                for joint in controlled_joints_names[1:]
            ],
        )

    def change_target(self, x_initial, target_position):
        self.output_queue.queue.clear()
        for i in range(3):
            self.oMtarget.translation[i] = target_position[i]

        self.crocoWrapper = OCP(self.param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(x_initial, self.oMtarget)

        for _ in range(self.delay):
            self.output_queue.put(
                (self.crocoWrapper.torque, x_initial, self.crocoWrapper.gain),
            )

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
        self.output_queue.put(
            (self.crocoWrapper.torque, self.x0, self.crocoWrapper.gain),
        )

        return self.output_queue.get()
