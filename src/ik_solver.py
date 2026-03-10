"""
Production IK Solver wrapper — clean API for the web app.
"""
import time
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_model import RobotModel
from src.model import create_model, IKNetV4, MODEL_REGISTRY
from src.utils import Normalizer, load_context_log


class IKSolver:
    """Neural Network Inverse Kinematics Solver."""

    def __init__(self, model_path=None, iteration=None):
        """
        Initialize the solver.

        Args:
            model_path: path to .pth model file. If None, auto-detect best.
            iteration: which architecture iteration. If None, auto-detect.
        """
        self.project_root = Path(__file__).parent.parent
        self.robot = RobotModel()
        self.device = torch.device('cpu')

        # Determine which model to load
        if model_path is None or iteration is None:
            ctx = load_context_log()
            best = ctx.get("best_model", {})
            if iteration is None:
                iteration = best.get("iteration", 1)
            if model_path is None:
                model_path = self.project_root / f"models/best_model_iter{iteration}.pth"

        self.iteration = iteration
        self.is_sincos = (iteration == 4)

        # Load model
        self.model, _ = create_model(iteration)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load normalizer
        self.normalizer = Normalizer()
        self.normalizer.load(str(self.project_root / "data" / "normalization_params.npz"))

        print(f"IK Solver ready (iteration {iteration})")

    def solve(self, target_pose):
        """
        Solve IK for a single target pose.

        Args:
            target_pose: [x, y, z, roll, pitch, yaw] (meters, radians)

        Returns:
            dict with joint_angles, position_error_mm, inference_time_ms
        """
        target_pose = np.asarray(target_pose, dtype=np.float32)

        # Normalize
        pose_norm = self.normalizer.normalize_input(target_pose.reshape(1, -1))
        x = torch.from_numpy(pose_norm.astype(np.float32)).to(self.device)

        # Predict
        start = time.perf_counter()
        with torch.no_grad():
            if self.is_sincos:
                pred_sincos = self.model(x)
                pred = torch.zeros(1, 6)
                for i in range(6):
                    pred[0, i] = torch.atan2(pred_sincos[0, 2*i], pred_sincos[0, 2*i+1])
                pred_norm = pred.numpy()
            else:
                pred_norm = self.model(x).numpy()

        inference_ms = (time.perf_counter() - start) * 1000

        # Denormalize
        joint_angles = self.normalizer.denormalize_output(pred_norm)[0]

        # Verify via FK
        try:
            achieved_pose = self.robot.forward_kinematics(joint_angles)
            pos_error_mm = np.linalg.norm(achieved_pose[:3] - target_pose[:3]) * 1000
            ori_error_deg = np.degrees(np.linalg.norm(achieved_pose[3:] - target_pose[3:]))
        except Exception:
            pos_error_mm = float('inf')
            ori_error_deg = float('inf')
            achieved_pose = np.zeros(6)

        return {
            "joint_angles": joint_angles.tolist(),
            "joint_angles_deg": np.degrees(joint_angles).tolist(),
            "achieved_pose": achieved_pose.tolist(),
            "position_error_mm": float(pos_error_mm),
            "orientation_error_deg": float(ori_error_deg),
            "inference_time_ms": float(inference_ms),
        }

    def solve_trajectory(self, waypoints):
        """
        Solve IK for a sequence of waypoints.

        Args:
            waypoints: (N, 6) array of target poses

        Returns:
            list of result dicts
        """
        results = []
        for wp in waypoints:
            results.append(self.solve(wp))
        return results

    def get_arm_positions(self, joint_angles):
        """Get 3D link positions for visualization."""
        return self.robot.get_link_positions(joint_angles).tolist()
