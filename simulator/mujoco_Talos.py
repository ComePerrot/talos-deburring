import mujoco
import mujoco.viewer
import numpy as np

from IPython import embed


class TalosMujoco:
    def __init__(self, controlled_joints_names):
        self.model = mujoco.MjModel.from_xml_path(
            "/home/cperrot/ws_bench/talos-limits/pal_talos/scene_motor.xml"
        )
        self.data = mujoco.MjData(self.model)

        self.viewer = None

        self.controlled_joints_names = controlled_joints_names
        self.controlled_joints_id = [
            self.model.joint(joint).id - 1 + 7 for joint in self.controlled_joints_names
        ]

        self.p_arm_gain = 100.0
        self.d_arm_gain = 8.0
        self.p_torso_gain = 500.0
        self.d_torso_gain = 20.0
        self.p_leg_gain = 800.0 * 1.2
        self.d_leg_gain = 35.0

        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 1)

        self.qpos_des = self.data.qpos.copy()
        self.qvel_des = self.data.qvel.copy()
        self.ctrl_ff = self.data.ctrl.copy()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        pass

    def pd_controller(self):
        for id in range(self.model.nu):
            act_name = self.data.actuator(id).name
            joint_name = act_name[:-7]
            joint_id = self.data.joint(joint_name).id

            d_pos = self.data.qpos[joint_id + 7 - 1] - self.qpos_des[joint_id + 7 - 1]
            d_vel = self.data.qvel[joint_id + 6 - 1] - self.qvel_des[joint_id + 6 - 1]

            if "torso" in joint_name:
                torque = (
                    self.ctrl_ff[id]
                    - self.p_torso_gain * d_pos
                    - self.d_torso_gain * d_vel
                )

            elif "arm" in joint_name:
                torque = (
                    self.ctrl_ff[id] - self.p_arm_gain * d_pos - self.d_arm_gain * d_vel
                )

            elif "leg" in joint_name:
                torque = (
                    self.ctrl_ff[id] - self.p_leg_gain * d_pos - self.d_leg_gain * d_vel
                )

            else:
                torque = 0

            self.data.ctrl[id] = torque


if __name__ == "__main__":
    controlled_joints = [
        "leg_left_1_joint",
        "leg_left_2_joint",
        "leg_left_3_joint",
        "leg_left_4_joint",
        "leg_left_5_joint",
        "leg_left_6_joint",
        "leg_right_1_joint",
        "leg_right_2_joint",
        "leg_right_3_joint",
        "leg_right_4_joint",
        "leg_right_5_joint",
        "leg_right_6_joint",
        "torso_1_joint",
        "torso_2_joint",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
    ]

    sim = TalosMujoco(controlled_joints_names=controlled_joints)

    while True:
        sim.pd_controller()
        # sim.render()
        sim.step()
        print(sim.data.qpos[:3])
