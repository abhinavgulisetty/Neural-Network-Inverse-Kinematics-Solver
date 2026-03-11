"""
Robot Model: PUMA 560 6-DOF manipulator with DH parameters, FK, and numerical IK.
Uses roboticstoolbox-python for reliable computations.
"""
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb


class RobotModel:
    """PUMA 560 6-DOF Industrial Manipulator."""

    def __init__(self):
        """Initialize PUMA 560 using roboticstoolbox built-in model."""
        self.robot = rtb.models.DH.Puma560()
        self.n_joints = 6

        self.joint_limits_lower_physical = np.array([-2.7925, -3.9270, -0.9425, -4.6426, -1.7453, -4.6426])
        self.joint_limits_upper_physical = np.array([2.7925, 0.7854, 3.0718, 4.6426, 1.7453, 4.6426])

        self.joint_limits_lower = np.radians([-90, -90,   0, -90, -90, -90])
        self.joint_limits_upper = np.radians([ 90,   0,  90,  90,  90,  90])

    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics for given joint angles.

        Args:
            joint_angles: array of 6 joint angles in radians

        Returns:
            pose: array [x, y, z, roll, pitch, yaw] (meters, radians)
        """
        q = np.asarray(joint_angles, dtype=np.float64)
        T = self.robot.fkine(q)

        pos = T.t
        rpy = T.rpy(order='zyx')

        return np.concatenate([pos, rpy])

    def batch_forward_kinematics(self, joint_angles_batch):
        """
        Compute FK for a batch of joint configurations.

        Args:
            joint_angles_batch: (N, 6) array of joint angles

        Returns:
            poses: (N, 6) array of [x, y, z, roll, pitch, yaw]
        """
        N = joint_angles_batch.shape[0]
        poses = np.zeros((N, 6))
        for i in range(N):
            poses[i] = self.forward_kinematics(joint_angles_batch[i])
        return poses

    def numerical_ik(self, target_pose, q0=None):
        """
        Solve IK numerically using roboticstoolbox.

        Args:
            target_pose: [x, y, z, roll, pitch, yaw]
            q0: initial joint angle guess (optional)

        Returns:
            joint_angles: solution (6,) or None if failed
            success: bool
            solve_time_ms: float
        """
        import time

        pos = target_pose[:3]
        rpy = target_pose[3:]

        T_target = SE3.Rt(SE3.RPY(rpy, order='zyx').R, pos)

        if q0 is None:
            q0 = np.zeros(self.n_joints)

        start = time.perf_counter()
        sol = self.robot.ikine_LM(T_target, q0=q0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if sol.success:
            return sol.q, True, elapsed_ms
        else:
            return None, False, elapsed_ms

    def get_link_positions(self, joint_angles):
        """
        Get 3D positions of all joints/links for visualization.

        Args:
            joint_angles: (6,) array of joint angles

        Returns:
            positions: (7, 3) array with base + 6 joint positions
        """
        q = np.asarray(joint_angles, dtype=np.float64)
        positions = [np.array([0, 0, 0])]

        for i in range(self.n_joints):
            try:
                T = self.robot.A(i, q)
                for j in range(i):
                    pass
                from functools import reduce
                Ts = [self.robot.links[j].A(q[j]) for j in range(i + 1)]
                T_cumulative = reduce(lambda a, b: a * b, Ts)
                positions.append(T_cumulative.t.copy())
            except Exception:
                T = self.robot.fkine(q)
                positions.append(T.t.copy())

        return np.array(positions)

    def random_joint_config(self, n=1):
        """Generate random joint configurations within restricted limits."""
        configs = np.random.uniform(
            self.joint_limits_lower,
            self.joint_limits_upper,
            size=(n, self.n_joints)
        )
        return configs if n > 1 else configs[0]

    def is_within_limits(self, joint_angles):
        """Check if joint angles are within restricted limits."""
        return np.all(joint_angles >= self.joint_limits_lower) and \
               np.all(joint_angles <= self.joint_limits_upper)


def verify_robot_model():
    """Run verification tests on the robot model."""
    print("=" * 60)
    print("ROBOT MODEL VERIFICATION")
    print("=" * 60)

    robot = RobotModel()

    print("\n--- Test 1: Home Position ---")
    home_q = np.zeros(6)
    home_pose = robot.forward_kinematics(home_q)
    print(f"Joint angles: {np.degrees(home_q).round(2)} deg")
    print(f"End-effector position: {home_pose[:3].round(4)} m")
    print(f"End-effector orientation: {np.degrees(home_pose[3:]).round(2)} deg")

    print("\n--- Test 2: Random Configuration ---")
    rand_q = robot.random_joint_config()
    rand_pose = robot.forward_kinematics(rand_q)
    print(f"Joint angles: {np.degrees(rand_q).round(2)} deg")
    print(f"End-effector position: {rand_pose[:3].round(4)} m")
    print(f"End-effector orientation: {np.degrees(rand_pose[3:]).round(2)} deg")

    print("\n--- Test 3: FK to IK to FK Round-Trip ---")
    test_q = np.array([0.5, -0.3, 0.8, 0.0, 0.5, 0.2])
    pose = robot.forward_kinematics(test_q)
    q_ik, success, solve_time = robot.numerical_ik(pose, q0=test_q + 0.01)
    if success:
        pose_recovered = robot.forward_kinematics(q_ik)
        pos_error = np.linalg.norm(pose[:3] - pose_recovered[:3]) * 1000
        print(f"Original pose: {pose[:3].round(4)} m")
        print(f"Recovered pose: {pose_recovered[:3].round(4)} m")
        print(f"Position error: {pos_error:.4f} mm")
        print(f"IK solve time: {solve_time:.2f} ms")
        print("Round-trip successful!" if pos_error < 1.0 else "Position error > 1mm")
    else:
        print("IK failed!")

    print("\n--- Test 4: Link Positions ---")
    link_pos = robot.get_link_positions(home_q)
    print(f"Number of link positions: {len(link_pos)}")
    for i, p in enumerate(link_pos):
        label = "Base" if i == 0 else f"Joint {i}"
        print(f"  {label}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m")

    print("\nRobot model verification complete!")
    return robot


if __name__ == "__main__":
    verify_robot_model()
