import example_robot_data
import numpy as np
import pinocchio as pin

pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)


class TalosDesigner:
    def __init__(
        self,
        URDF,
        SRDF,
        toolPosition,
        controlledJoints,
        set_gravity,
        dt,
        **kwargs,
    ):
        modelPath = example_robot_data.getModelPath(URDF)
        self.URDF_path = modelPath + URDF
        self.gravity = np.array([0, 0, -9.81]) if set_gravity else np.array([0, 0, 0])
        self.dt = dt
        if True:
            self.rmodelComplete = pin.buildModelFromUrdf(
                self.URDF_path,
                pin.JointModelFreeFlyer(),
            )
        else:
            self.rmodelComplete = pin.buildModelFromUrdf(self.URDF_path)

        self._refineModel(self.rmodelComplete, SRDF)
        self._addLimits()

        self._addTool(toolPosition)
        # self._addshoulder()

        self._buildReducedModel(controlledJoints)

    def _refineModel(self, model, SRDF):
        """Load additional information from SRDF file
        rotor inertia and gear ratio

        :param model Model of the robot to refine
        :param SRDF Path to SRDF file containing data to add to model
        """
        modelPath = example_robot_data.getModelPath(SRDF)

        pin.loadRotorParameters(model, modelPath + SRDF, False)
        model.armature = np.multiply(
            model.rotorInertia.flat,
            np.square(model.rotorGearRatio.flat),
        )

        pin.loadReferenceConfigurations(model, modelPath + SRDF, False)

    def _addLimits(self):
        """Add free flyers joint limits"""
        self.rmodelComplete.upperPositionLimit[:7] = 1
        self.rmodelComplete.lowerPositionLimit[:7] = -1

    def _addTool(self, toolPosition):
        """Add frame corresponding to the tool

        :param toolPosition Position of the tool frame in parent frame
        """
        placement_tool = pin.SE3.Identity()
        placement_tool.translation[0] = toolPosition[0]
        placement_tool.translation[1] = toolPosition[1]
        placement_tool.translation[2] = toolPosition[2]

        self.rmodelComplete.addBodyFrame(
            "driller",
            self.rmodelComplete.getJointId("gripper_left_joint"),
            placement_tool,
            self.rmodelComplete.getFrameId("gripper_left_fingertip_3_link"),
        )

        self.endEffectorId = self.rmodelComplete.getFrameId("driller")

    def _buildReducedModel(self, controlledJointsName):
        """Build a reduce model for which only selected joints are controlled

        :param controlledJoints List of the joints to control
        """
        self.q0Complete = self.rmodelComplete.referenceConfigurations["half_sitting"]
        self.z_c = self.q0Complete[2]
        self.g = self.gravity[2]
        self.base_translation_bullet_pinocchio = np.array([-0.08222, 0.00838, -0.07261])
        self.base_robot_bullet_SE3_origin_robot_pinocchio = pin.SE3(
            np.eye(3),
            self.q0Complete[:3] + self.base_translation_bullet_pinocchio,
        ).inverse()
        # Check that controlled joints belong to model
        for joint in controlledJointsName:
            if joint not in self.rmodelComplete.names:
                print("ERROR")

        self.controlledJointsID = [
            i
            for (i, n) in enumerate(self.rmodelComplete.names)
            if n in controlledJointsName
        ]

        # Make list of blocked joints
        lockedJointsID = [
            self.rmodelComplete.getJointId(joint)
            for joint in self.rmodelComplete.names[1:]
            if joint not in controlledJointsName
        ]

        self.rmodel = pin.buildReducedModel(
            self.rmodelComplete,
            lockedJointsID,
            self.q0Complete,
        )
        self.rdata = self.rmodel.createData()

        # Define a default State
        self.q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])

    def update_reduced_model(self, x_measured, external_forces, base_pos):
        pin.forwardKinematics(
            self.rmodel,
            self.rdata,
            x_measured[: self.rmodel.nq],
            x_measured[-self.rmodel.nv :],
        )
        pin.updateFramePlacements(self.rmodel, self.rdata)

        # Updating from bullet world to base robot bullet
        self.world_bullet_SE3_base_robot_bullet = pin.XYZQUATToSE3(base_pos)
        self.world_bullet_SE3_origin_robot_pin = (
            self.world_bullet_SE3_base_robot_bullet
            * self.base_robot_bullet_SE3_origin_robot_pinocchio
        )
        self._calculate_CoM(x_measured)
        self._calculate_ZMP(external_forces)

        self.oMtool = self.rdata.oMf[self.endEffectorId]

    def _calculate_CoM(self, x_measured):
        """Compute the CoM position from the robot state"""
        local_CoM = pin.centerOfMass(
            self.rmodel,
            self.rdata,
            x_measured[: self.rmodel.nq],
        )
        self._CoM = (
            self.world_bullet_SE3_origin_robot_pin.rotation @ local_CoM
            + self.world_bullet_SE3_origin_robot_pin.translation
        )

    def _calculate_ZMP(self, external_forces):
        """Compute the ZMP position from the robot state"""
        try:
            sum_tot = 0
            sum_px = 0
            sum_py = 0
            for i in range(len(external_forces)):
                sum_px += external_forces[i][1][0] * external_forces[i][2]
                sum_py += external_forces[i][1][1] * external_forces[i][2]
                sum_tot += external_forces[i][2]
            ZMP = np.array([sum_px / sum_tot, sum_py / sum_tot, 0])
            self._ZMP = ZMP
        except:  # noqa: E722
            self._ZMP = self._CoM

    def get_end_effector_pos(self):
        """Compute the end effector position from the robot state"""
        return (
            self.world_bullet_SE3_origin_robot_pin.rotation @ self.oMtool.translation
            + self.world_bullet_SE3_origin_robot_pin.translation
        )

    def get_CoM(self):
        """Return the CoM position from the robot state"""
        return self._CoM

    def get_ZMP(self):
        """Return the ZMP position from the robot state"""
        return self._ZMP
