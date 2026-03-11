import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import Normalizer
from src.robot_model import RobotModel
from src.model import IKNetV1, IKNetV3

def test_predictions():
    data_dir = project_root / "data"
    
    # Load dataset
    data = np.load(data_dir / "test.npz")
    poses = data['poses']
    joints = data['joint_angles']
    
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    poses_norm = normalizer.normalize_input(poses)
    
    # Check bounds
    print("Poses stats:")
    print("Min:", poses.min(axis=0))
    print("Max:", poses.max(axis=0))
    print("\nPoses Norm stats:")
    print("Min:", poses_norm.min(axis=0))
    print("Max:", poses_norm.max(axis=0))
    
    # Find two samples with very similar poses but different joints
    print("\nLooking for similar poses with different joint angles...")
    # Just take the first pose and find closest poses
    p0 = poses[0]
    dists = np.linalg.norm(poses[:, :3] - p0[:3], axis=1)
    
    # Sort by distance
    idx = np.argsort(dists)
    
    print(f"Base pose: {p0}")
    print(f"Base joints (deg): {np.degrees(joints[idx[0]])}")
    
    for i in range(1, 10):
        j = idx[i]
        j_dist = np.linalg.norm(joints[idx[0]] - joints[j])
        if j_dist > 1.0: # Different joint configuration
            print(f"Found nearby pose (dist {dists[j]:.4f}): {poses[j]}")
            print(f"Joints (deg): {np.degrees(joints[j])}")
            print(f"Joint dist: {j_dist:.4f} rad")
            break

if __name__ == "__main__":
    test_predictions()
