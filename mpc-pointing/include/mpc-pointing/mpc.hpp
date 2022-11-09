#ifndef MPC_P
#define MPC_P

#include "mpc-pointing/fwd.hpp"
#include "mpc-pointing/ocp.hpp"

namespace mpc_p {

struct MPCSettings_Point {
  // Timing parameters
  size_t T_initialization;
  size_t T_stabilization;
  size_t T_drilling;

  // Mocap
  int use_mocap;

  // GainScheduling
  int use_gainScheduling;
  double gainSchedulig_slope;
  double maxGoalWeight;

  // Target
  Vector3d targetPos;
  std::vector<Vector3d> holes_offsets;
  double backwardOffset;
  double tolerance;

  void readParamsFromYamlString(std::string &StringToParse);
  void readParamsFromYamlFile(const std::string &Filename);
};

class MPC_Point {
 private:
  MPCSettings_Point settings_;
  RobotWrapper designer_;
  OCP_Point OCP_;

  VectorXd x0_;
  VectorXd u0_;
  MatrixXd K0_;

  // MPC State
  size_t current_hole_ = 0;
  int drilling_state_ = 0;
  size_t iteration_ = 0;
  double goal_weight_ = 0;

  // Target related variables
  size_t number_holes_;
  std::vector<SE3> holes_offsets_;
  std::vector<SE3> list_oMhole_;  // Holes position in the robot frame
  SE3 backwardOffset_ = SE3::Identity();

  // Security management
  bool initialized_ = false;

  // Memory preallocations:
  SE3 oMtarget_;
  SE3 oMbackwardHole_;
  SE3 tool_se3_hole_;
  double position_error_ = 0;
  std::vector<unsigned long> controlled_joints_id_;
  VectorXd x_internal_;

 private:
  void setTarget(const SE3 &toolMtarget);
  void setHolesPlacement();
  void updateTarget(const SE3 &toolMtarget);
  void updateOCP();

 public:
  MPC_Point(const MPCSettings_Point &settings,
            const OCPSettings_Point &OCPSettings, const RobotWrapper &designer);

  void initialize(const ConstVectorRef &q0, const ConstVectorRef &v0,
                  const SE3 &toolMtarget);

  void iterate(const VectorXd &x0, const SE3 &toolMtarget);

  void iterate(const ConstVectorRef &q_current, const ConstVectorRef &v_current,
               const SE3 &toolMtarget);

  const VectorXd &shapeState(const ConstVectorRef &q, const ConstVectorRef &v);

  // getters and setters
  MPCSettings_Point &get_settings() { return settings_; }

  const VectorXd &get_x0() const { return x0_; }

  const VectorXd &get_u0() const { return u0_; }

  const MatrixXd &get_K0() const { return K0_; }

  const pinocchio::SE3 &get_Target_frame() const { return oMtarget_; }

  OCP_Point &get_OCP() { return OCP_; }
  void set_OCP(const OCP_Point &OCP) { OCP_ = OCP; }

  RobotWrapper &get_designer() { return designer_; }
  void set_designer(const RobotWrapper &designer) { designer_ = designer; }
};
}  // namespace mpc_p

#endif  // MPC_P
