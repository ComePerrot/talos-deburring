#!/usr/bin/env python

# import sys
import unittest
from time import sleep

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from linear_feedback_controller_msgs.msg import Control, Sensor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

PKG = "deburring_ros_interface"
NAME = "MPC_Interface_test"


class TestMPCInterface(unittest.TestCase):
    def test_message_reception(self):
        rospy.init_node(NAME, anonymous=True)

        self.sensorMsg = self._define_sensor_msg()

        print("Publishing message")
        pub = rospy.Publisher(
            "/linear_feedback_controller/sensor_state", Sensor, queue_size=10
        )
        for _ in range(5):
            pub.publish(self.sensorMsg)
            sleep(1)

        print("Waiting for answer")
        msg = rospy.wait_for_message(
            "/linear_feedback_controller/desired_control", Control, timeout=30
        )
        self.assertEquals(
            msg.initial_state.base_pose,
            self.sensorMsg.base_pose,
            "Base pose in received initial state does not match the one sent to the robot",
        )
        self.assertEquals(
            msg.initial_state.base_twist,
            self.sensorMsg.base_twist,
            "Base twist in received initial state does not match the one sent to the robot",
        )

    def _define_sensor_msg(self):
        controlled_joints = rospy.get_param("controlled_joints")

        sensorData = Sensor(
            header=Header(stamp=rospy.get_rostime()),
            base_pose=Pose(
                position=Point(x=0, y=0, z=1.01927),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            base_twist=Twist(
                linear=Vector3(x=0, y=0, z=0), angular=Vector3(x=0, y=0, z=0)
            ),
            joint_state=JointState(
                name=controlled_joints,
                position=np.ones(len(controlled_joints) - 1),
                velocity=np.zeros(len(controlled_joints) - 1),
            ),
        )

        return sensorData


if __name__ == "__main__":
    import rostest

    rostest.rosrun(PKG, NAME, TestMPCInterface)
