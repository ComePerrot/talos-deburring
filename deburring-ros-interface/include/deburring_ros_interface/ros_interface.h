#ifndef DEBURRING_ROS_INTERFACE
#define DEBURRING_ROS_INTERFACE

#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/TransformStamped.h>
#include <linear_feedback_controller_msgs/Control.h>
#include <linear_feedback_controller_msgs/Sensor.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/ros.h>

#include <Eigen/Core>
#include <linear_feedback_controller_msgs/eigen_conversions.hpp>
#include <map>
#include <string>

class DeburringROSInterface {
 public:
  DeburringROSInterface(ros::NodeHandle nh);
  void update(const Eigen::VectorXd& u0, const Eigen::MatrixXd& K0,
              const Eigen::VectorXd& wrench_left = Eigen::VectorXd::Zero(6),
              const Eigen::VectorXd& wrench_right = Eigen::VectorXd::Zero(6));

  Eigen::VectorXd& get_robotState();

 private:
  void SensorCb(const linear_feedback_controller_msgs::SensorConstPtr& msg);
  void mapMsgToJointSates();
  void mapControlToMsg(const Eigen::VectorXd& u0, const Eigen::MatrixXd& K0);
  void mapWrenchesToMsg(const Eigen::VectorXd& wrench_left,
                        const Eigen::VectorXd& wrench_right);

  ros::Subscriber sensor_sub_;
  boost::shared_ptr<realtime_tools::RealtimePublisher<
      linear_feedback_controller_msgs::Control>>
      command_pub_;

  // Ros messages
  linear_feedback_controller_msgs::Sensor sensor_msg_;
  linear_feedback_controller_msgs::Control control_msg_;

  // Robot state (with joints)
  Eigen::VectorXd joint_states_;

  // prealocated memory
  linear_feedback_controller_msgs::Eigen::Sensor sensor_eigen_;
};

#endif  // DEBURRING_ROS_INTERFACE
