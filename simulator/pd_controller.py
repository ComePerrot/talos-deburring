class PDController:
    def __init__(self):
        self.p_arm_gain = 100.0
        self.d_arm_gain = 8.0
        self.p_torso_gain = 500.0
        self.d_torso_gain = 20.0
        self.p_leg_gain = 800.0
        self.d_leg_gain = 35.0
        self.initial_joint_positions = {
            "leg_left_1_joint": 0,
            "leg_left_2_joint": 0,
            "leg_left_3_joint": -0.4,
            "leg_left_4_joint": 0.8,
            "leg_left_5_joint": -0.4,
            "leg_left_6_joint": 0,
            "leg_right_1_joint": 0,
            "leg_right_2_joint": 0,
            "leg_right_3_joint": -0.4,
            "leg_right_4_joint": 0.8,
            "leg_right_5_joint": -0.4,
            "leg_right_6_joint": 0,
            "torso_1_joint": 0,
            "torso_2_joint": 0,
            "arm_left_1_joint": 0.4,
            "arm_left_2_joint": 0.24,
            "arm_left_3_joint": -0.6,
            "arm_left_4_joint": -1.45,
            "arm_left_5_joint": 0,
            "arm_left_6_joint": 0,
            "arm_left_7_joint": 0,
            "arm_right_1_joint": -0.4,
            "arm_right_2_joint": -0.24,
            "arm_right_3_joint": 0.6,
            "arm_right_4_joint": -1.45,
            "arm_right_5_joint": 0,
            "arm_right_6_joint": 0,
            "arm_right_7_joint": 0,
        }
        self.feed_forward = {
            "leg_left_1_joint": 0,
            "leg_left_2_joint": 1,
            "leg_left_3_joint": 2,
            "leg_left_4_joint": -5e01,
            "leg_left_5_joint": 3,
            "leg_left_6_joint": -5,
            "leg_right_1_joint": 0,
            "leg_right_2_joint": 1,
            "leg_right_3_joint": 2,
            "leg_right_4_joint": -5e01,
            "leg_right_5_joint": 3,
            "leg_right_6_joint": 5,
            "torso_1_joint": 0,
            "torso_2_joint": 6e-1,
            "arm_left_1_joint": 6e-02,
            "arm_left_2_joint": 5e-01,
            "arm_left_3_joint": 2.2,
            "arm_left_4_joint": -9.3,
            "arm_left_5_joint": 1.1e-01,
            "arm_left_6_joint": 3.2e-01,
            "arm_left_7_joint": -1.5,
            "arm_right_1_joint": -6e-02,
            "arm_right_2_joint": -5e-01,
            "arm_right_3_joint": -2.2,
            "arm_right_4_joint": -9.3,
            "arm_right_5_joint": 1.1e-01,
            "arm_right_6_joint": 3.2e-01,
            "arm_right_7_joint": -1.5,
        }

    def compute_control(self, joint_name, measured_pos, measured_vel):
        feed_forward = self.feed_forward[joint_name]
        d_pos = measured_pos - self.initial_joint_positions[joint_name]
        d_vel = measured_vel

        print(measured_pos)
        print(measured_vel)

        if "torso" in joint_name:
            torque = (
                feed_forward - self.p_torso_gain * d_pos - self.d_torso_gain * d_vel
            )

        elif "arm" in joint_name:
            torque = feed_forward - self.p_arm_gain * d_pos - self.d_arm_gain * d_vel

        elif "leg" in joint_name:
            torque = feed_forward - self.p_leg_gain * d_pos - self.d_leg_gain * d_vel

        return torque
