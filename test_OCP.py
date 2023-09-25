#!/usr/bin/env python

#####################
#  LOADING MODULES  #
#####################

import pinocchio as pin
import numpy as np
import yaml

from deburring_mpc import OCP, RobotDesigner

from simulator.bullet_Talos import TalosDeburringSimulator
from utils.plotter import TalosPlotter

################
#  PARAMETERS  #
################

enableGUI = True
plotResults = False
plotCosts = False

targetPos_1 = [0.6, 0.4, 1.1]
targetTolerance = 0.005

# Timing settings
T_total = 400
T_init = 0

# Variable Posture
variablePosture = False
ref_leftArm = [
    -0.08419471,
    0.425144,
    0.00556666,
    -1.50516856,
    # 0.68574977,
    # 0.18184998,
    # -0.07185897,
]

modelPath = "/opt/openrobots/share/example-robot-data/robots/talos_data/"
URDF = modelPath + "robots/talos_reduced.urdf"
SRDF = modelPath + "srdf/talos.srdf"

# Parameters filename
filename = "settings_sobec.yaml"

####################
#  INITIALIZATION  #
####################

# Loading extra parameters from file
with open(filename, "r") as paramFile:
    params = yaml.safe_load(paramFile)

controlledJoints = params["robot"]["controlledJoints"]
toolFramePos = params["robot"]["toolFramePos"]
OCPparams = params["OCP"]

OCPparams["time_step"] = 0.01
OCPparams["state_weights"] = np.array(
    [
        500,
        500,
        500,
        1000,
        1000,
        1000,
        500,
        500,
        500,
        500,
        1000,
        1000,
        500,
        500,
        500,
        500,
        1000,
        1000,
        100,
        200,
        100,
        100,
        100,
        100,
        1,
        1,
        1,
        500,
        500,
        500,
        500,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    ]
)
OCPparams["control_weights"] = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
)

horizonLength = OCPparams["horizon_length"]

# Robot model
design_conf = dict(
    urdf_path=URDF,
    srdf_path=SRDF,
    left_foot_name="right_sole_link",
    right_foot_name="left_sole_link",
    robot_description="",
    controlled_joints_names=controlledJoints,
)
pinWrapper = RobotDesigner()
pinWrapper.initialize(design_conf)

gripper_SE3_tool = pin.SE3.Identity()
gripper_SE3_tool.translation[0] = toolFramePos[0]
gripper_SE3_tool.translation[1] = toolFramePos[1]
gripper_SE3_tool.translation[2] = toolFramePos[2]

pinWrapper.add_end_effector_frame(
    "deburring_tool", "gripper_left_fingertip_3_link", gripper_SE3_tool
)

rModel = pinWrapper.get_rmodel()

rModel.lowerPositionLimit = np.array(
    [
        # Base
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        # Left leg
        -0.35,
        -0.52,
        -2.10,
        0.0,
        -1.31,
        -0.52,
        # Right leg
        -1.57,
        -0.52,
        -2.10,
        0.0,
        -1.31,
        -0.52,
        # Torso
        -1.3,
        -0.1,
        # Left arm
        -1.57,
        0.2,
        -2.44,
        -2.1,
        -2.53,
        -1.3,
        -0.6,
        # Right arm
        -0.4,
        -2.88,
        -2.44,
        -2,
    ]
)

rModel.upperPositionLimit = np.array(
    [
        # Base
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        # Left leg
        1.57,
        0.52,
        0.7,
        2.62,
        0.77,
        0.52,
        # Right leg
        0.35,
        0.52,
        0.7,
        2.62,
        0.77,
        0.52,
        # Torso
        1.3,
        0.78,
        # Left arm
        0.52,
        2.88,
        2.44,
        0,
        2.53,
        1.3,
        0.6,
        # Right arm
        1.57,
        -0.2,
        2.44,
        0,
    ]
)

# OCP
oMtarget = pin.SE3.Identity()
oMtarget.translation[0] = targetPos_1[0]
oMtarget.translation[1] = targetPos_1[1]
oMtarget.translation[2] = targetPos_1[2]

oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

OCP = OCP(OCPparams, pinWrapper)
OCP.initialize(pinWrapper.get_x0(), oMtarget)
print("OCP successfully loaded")

ddp = OCP.solver

# Simulator
simulator = TalosDeburringSimulator(
    URDF=URDF,
    initialConfiguration=pinWrapper.get_q0_complete(),
    robotJointNames=pinWrapper.get_rmodel_complete().names,
    controlledJointsIDs=pinWrapper.get_controlled_joints_ids(),
    toolPlacement=pinWrapper.get_end_effector_frame(),
    targetPlacement=oMtarget,
    enableGUI=enableGUI,
)

# Plotter
plotter = TalosPlotter(pinWrapper.get_rmodel(), T_total)

########################
#  INITIAL RESOLUTION  #
########################

for index in range(horizonLength):
    OCP.change_goal_cost_activation(index, True)

x_measured = simulator.getRobotState()
OCP.solve_first(pinWrapper.get_x0())

# Serialize initial resolution data
xs_init = ddp.xs.tolist()
us_init = ddp.us.tolist()


# plot_state_from_dic(return_state_vector(OCP.solver))

# ref_leftArm = OCP.solver.xs[-1][7 + 14 : 7 + 18]

for index in range(horizonLength):
    OCP.change_goal_cost_activation(index, False)
OCP.solve_first(pinWrapper.get_x0())

# Serialize initial resolution data
xs_init = ddp.xs.tolist()
us_init = ddp.us.tolist()

###############
#  MAIN LOOP  #
###############
NcontrolKnots = 10
T = 0
drilling_state = 0
targetReached = 0
reachTime = 0

state = OCP.state
toolPlacement = pinWrapper.get_end_effector_frame()
ddp = OCP.solver

goalTrackingWeight = 10

x_ref = pinWrapper.get_x0().copy()
x_ref[7 + 14 : 7 + 18] = ref_leftArm

for T in range(T_total):
    # Controller works faster than trajectory generation
    for i_control in range(NcontrolKnots):
        if i_control == 0:
            x0 = simulator.getRobotState()

        x_measured = simulator.getRobotState()

        # Compute torque to be applied by adding Riccati term
        torques = OCP.torque + OCP.gain @ state.diff(x_measured, x0)

        # Apply torque on complete model
        simulator.step(torques, toolPlacement, oMtarget)

    # Solve MPC iteration
    pinWrapper.update_reduced_model(x_measured)
    OCP.solve(x_measured)

    # Update tool placement
    toolPlacement = pinWrapper.get_end_effector_frame()

    # Handling phases of the movement
    if T < T_init:  # Initial phase
        drilling_state = 0
        pass
    else:  # Approach phase
        drilling_state = 1
        index = horizonLength + T_init - T
        # OCP.change_goal_cost_activation(index, True)

        if variablePosture:
            OCP.recede()
            OCP.change_goal_cost_activation(horizonLength - 1, True)
            for i in range(horizonLength):
                OCP.change_posture_reference(i, x0)
        pass

    # Checking if target is reached
    errorPlacement = np.linalg.norm(toolPlacement.translation - oMtarget.translation)

    if errorPlacement < targetTolerance:
        if not targetReached:
            targetReached = True
            reachTime = T - T_init
    else:
        targetReached = False

    # Log robot data
    plotter.logDrillingState(T, drilling_state)
    plotter.logState(T, x_measured)
    # plotter.logTorques(T, torques)
    plotter.logEndEffectorPos(T, toolPlacement.translation, oMtarget.translation)
    plotter.logTargetReached(T, targetReached)

    pin.forwardKinematics(
        pinWrapper.get_rmodel(),
        pinWrapper.get_rdata(),
        x_measured[: pinWrapper.get_rmodel().nq],
        x_measured[-pinWrapper.get_rmodel().nv :],
    )
    speed = pin.getFrameVelocity(
        pinWrapper.get_rmodel(),
        pinWrapper.get_rdata(),
        pinWrapper.get_end_effector_id(),
    ).linear

    plotter.logEndEffectorSpeed(T, speed)

    # Check stop condition
    if toolPlacement.translation[1] > 1 or toolPlacement.translation[1] < 0:
        break

simulator.end
print("Simulation ended")

# Reporting performances
print("Target reached: " + str(targetReached))
print("Error=" + str(errorPlacement / 1e-3) + " mm")
print("Reach Time=" + str(reachTime / 100) + " s")

if plotResults:
    plotter.plotResults()
