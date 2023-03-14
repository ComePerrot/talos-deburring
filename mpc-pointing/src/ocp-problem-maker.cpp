#include "mpc-pointing/ocp.hpp"

namespace mpc_p {
void OCP_Point::buildSolver(const VectorXd x0, SE3 oMtarget) {
  designer_.updateReducedModel(x0);

  state_ = boost::make_shared<crocoddyl::StateMultibody>(
      boost::make_shared<pinocchio::Model>(designer_.get_rModel()));
  actuation_ =
      boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state_);

  auto runningModels = std::vector<ActionModel>(settings_.horizon_length);

  for (size_t i = 0; i < settings_.horizon_length; i++) {
    runningModels[i] = formulatePointingTask();
  }

  // Terminal model
  auto terminalModel = formulatePointingTask();

  boost::shared_ptr<crocoddyl::ShootingProblem> shooting_problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                     terminalModel);
  solver_ = boost::make_shared<crocoddyl::SolverFDDP>(shooting_problem);

  // Change References
  VectorXd postureReference = x0;
  postureReference.tail(designer_.get_rModel().nv) =
      VectorXd::Zero(designer_.get_rModel().nv);
  for (size_t modelIndex = 0; modelIndex <= settings_.horizon_length;
       modelIndex++) {
    changePostureReference(modelIndex, postureReference);
  }
  setBalancingTorques();

  // Change Target placemenet
  updateGoalPosition(oMtarget.translation());
  updateGoalRotation(oMtarget.rotation());

  // Deactivate Target cost at the beginning
  for (size_t modelIndex = 0; modelIndex <= settings_.horizon_length;
       modelIndex++) {
    changeGoalCostActivation(modelIndex, false);
  }
}

void OCP_Point::solveFirst(const VectorXd x) {
  // horizon settings
  std::vector<VectorXd> xs_init;
  std::vector<VectorXd> us_init;

  for (std::size_t i = 0; i < settings_.horizon_length; i++) {
    xs_init.push_back(x);
    // Gravity compensation torques
    us_init.push_back(
        boost::static_pointer_cast<crocoddyl::ResidualModelControl>(
            costs(i)->get_costs().at("actuationTask")->cost->get_residual())
            ->get_reference());
  }
  xs_init.push_back(x);

  solver_->solve(xs_init, us_init, 500, false);
}

ActionModel OCP_Point::formulatePointingTask() {
  Contact contacts = boost::make_shared<crocoddyl::ContactModelMultiple>(
      state_, actuation_->get_nu());
  CostModelSum costs =
      boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());

  defineFeetContact(contacts);

  // Safety constraints
  defineJointLimits(costs, settings_.wLimit, settings_.scaleLimits);

  // Equilibrium constraints
  defineCoMPosition(costs, settings_.wPCoM);

  // Regulation task
  definePostureTask(costs, settings_.wStateReg);
  defineActuationTask(costs, settings_.wControlReg);

  // End effector task
  defineGripperPlacement(costs, settings_.wGripperPos, settings_.wGripperRot);
  defineGripperVelocity(costs, settings_.wGripperVel);

  DifferentialActionModel runningDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state_, actuation_, contacts, costs, 0., true);

  auto pin_model_ = designer_.get_rModel();
  VectorXd armature_ = Eigen::VectorXd::Zero(pin_model_.nv);
  armature_[(long)pin_model_.getJointId("arm_left_5_joint") + 4] = 0.1;  // 0.7
  armature_[(long)pin_model_.getJointId("arm_left_6_joint") + 4] = 0.1;  // 0.7
  armature_[(long)pin_model_.getJointId("arm_left_7_joint") + 4] = 0.1;  // 1
  runningDAM->set_armature(armature_);
  ActionModel runningModel =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
          runningDAM, settings_.timeStep);

  return runningModel;
}

}  // namespace mpc_p