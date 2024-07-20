import numpy as np


class LimitChecker:
    """A class to check if a given set of (state, command) infringes the limits of Talos

    This class provides functionality to easily check if a command or a state exceeds the limits of Talos.
    There are the checked limits:
        - position
        - velocity
        - effort

    Args:
        rmodel: The Pinocchio model of the robot.
        verbose: If True, additional debug information such as joint ID, current value, and limit
                will be returned when a limit is broken. Defaults to False.
    """

    def __init__(self, rmodel, verbose=False):
        """Defines the Limit Checker class

        The limits are copied from the Pinocchio model (i.e. extracted from the URDF).
        Some limits are modified using custom values.
        """
        self.verbose = verbose
        self.rmodel = rmodel

        # Check if the robot is fixed base or free-flyer
        self.has_free_flyer = "root_joint" in rmodel.names

        # Remove universe and root_joint (if applicable) from the joint names
        self.joint_names = rmodel.names[1 + int(self.has_free_flyer) :].tolist()
        self.nq = len(self.joint_names)

        # Load limits from model
        self.upper_pos_limits = rmodel.upperPositionLimit[
            7 * int(self.has_free_flyer) :
        ].copy()
        self.lower_pos_limits = rmodel.lowerPositionLimit[
            7 * int(self.has_free_flyer) :
        ].copy()
        self.velocity_limits = rmodel.velocityLimit[
            6 * int(self.has_free_flyer) :
        ].copy()
        self.effort_limits = rmodel.effortLimit[6 * int(self.has_free_flyer) :].copy()

    def are_limits_broken(self, q, q_dot, u):
        """Check all the limits of the robots.

        Returns False if no limit is broken.

        Args:
            q: joint position (without free-flyer)
            q_dot: joint velocity (without free-flyer)
            u: joint effort
        """
        assert (
            len(q) == self.nq
        ), f"Size mismatch: q is of size ({len(q)}) instead of ({self.nq})."
        assert (
            len(q_dot) == self.nq
        ), f"Size mismatch: q_dot is of size ({len(q_dot)}) instead of ({self.nq})."
        assert (
            len(u) == self.nq
        ), f"Size mismatch: u is of size ({len(u)}) instead of ({self.nq})."

        position_limits = [
            self.check_position_limit(q_i, i)
            for i, q_i in enumerate(q)
            if self.check_position_limit(q[i], i)
        ]
        velocity_limits = [
            self.check_velocity_limit(q_dot_i, i)
            for i, q_dot_i in enumerate(q_dot)
            if self.check_velocity_limit(q_dot_i, i)
        ]
        effort_limits = [
            self.check_effort_limit(u_i, i)
            for i, u_i in enumerate(u)
            if self.check_effort_limit(u_i, i)
        ]

        limits = {
            "position": position_limits,
            "velocity": velocity_limits,
            "effort": effort_limits,
        }

        if any(len(limit) > 0 for limit in limits.values()):
            if self.verbose:
                return limits
            return True
        return False

    def check_position_limit(self, position, joint_id):
        """Check if position limits are broken for one joint

        Args:
            position: position of the joint
            joint_id: id of the joint to check
        """
        if position > self.upper_pos_limits[joint_id]:
            return (
                self.joint_names[joint_id],
                (position, self.upper_pos_limits[joint_id]),
            )
        if position < self.lower_pos_limits[joint_id]:
            return (
                self.joint_names[joint_id],
                (position, self.lower_pos_limits[joint_id]),
            )
        return None

    def check_velocity_limit(self, velocity, joint_id):
        """Check if velocity limits are broken for one joint

        Args:
            position: velocity of the joint
            joint_id: id of the joint to check
        """
        if np.abs(velocity) > self.velocity_limits[joint_id]:
            return (
                self.joint_names[joint_id],
                (velocity, self.velocity_limits[joint_id]),
            )
        return None

    def check_effort_limit(self, effort, joint_id):
        """Check if effort limits are broken for one joint

        Args:
            position: effort in the joint
            joint_id: id of the joint to check
        """
        if np.abs(effort) > self.effort_limits[joint_id]:
            return (
                self.joint_names[joint_id],
                (effort, self.effort_limits[joint_id]),
            )
        return None
