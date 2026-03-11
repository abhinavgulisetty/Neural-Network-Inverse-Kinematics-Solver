"""
Training Data Generator: Uniform + Singularity + Boundary sampling.
"""
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_model import RobotModel
from src.utils import Normalizer, log_dataset_stats, update_phase, ensure_dir


class DataGenerator:
    """Generate training data using forward kinematics."""

    def __init__(self):
        self.robot = RobotModel()
        self.data_dir = Path(__file__).parent.parent / "data"
        ensure_dir(self.data_dir)

    def generate_uniform(self, n_samples=100000):
        """Phase 2a: Uniform random sampling across joint space."""
        print(f"\n=== Generating {n_samples} uniform samples ===")
        joint_angles = self.robot.random_joint_config(n=n_samples)
        poses = np.zeros((n_samples, 6))

        valid_mask = np.ones(n_samples, dtype=bool)
        for i in tqdm(range(n_samples), desc="Forward Kinematics"):
            try:
                poses[i] = self.robot.forward_kinematics(joint_angles[i])
                if np.any(np.isnan(poses[i])) or np.any(np.isinf(poses[i])):
                    valid_mask[i] = False
            except Exception:
                valid_mask[i] = False

        # Filter valid samples
        joint_angles = joint_angles[valid_mask]
        poses = poses[valid_mask]
        print(f"Valid samples: {len(joint_angles)}/{n_samples}")

        np.savez(self.data_dir / "dataset_uniform.npz",
                 poses=poses, joint_angles=joint_angles)
        return poses, joint_angles

    def generate_singularity(self, n_samples=50000):
        """Phase 2b: Extra samples near singularity configurations."""
        print(f"\n=== Generating {n_samples} singularity-region samples ===")
        joint_angles = np.zeros((n_samples, 6))
        n_per_type = n_samples // 3

        # Shoulder singularity: theta2 near 0 or ±pi. Restricted to [-90, 0] so only 0 is possible.
        for i in range(n_per_type):
            q = self.robot.random_joint_config()
            q[1] = np.random.normal(0, 0.1)  # Only 0 is in our restricted range
            q = np.clip(q, self.robot.joint_limits_lower, self.robot.joint_limits_upper)
            joint_angles[i] = q

        # Elbow singularity: theta3 near 0 or ±pi. Restricted to [0, 90] so only 0 is possible.
        for i in range(n_per_type, 2 * n_per_type):
            q = self.robot.random_joint_config()
            q[2] = np.random.normal(0, 0.1)  # Only 0 is in our restricted range
            q = np.clip(q, self.robot.joint_limits_lower, self.robot.joint_limits_upper)
            joint_angles[i] = q

        # Wrist singularity: theta5 near 0
        for i in range(2 * n_per_type, n_samples):
            q = self.robot.random_joint_config()
            q[4] = np.random.normal(0, 0.1)
            q = np.clip(q, self.robot.joint_limits_lower, self.robot.joint_limits_upper)
            joint_angles[i] = q

        poses = np.zeros((n_samples, 6))
        valid_mask = np.ones(n_samples, dtype=bool)
        for i in tqdm(range(n_samples), desc="FK (singularity)"):
            try:
                poses[i] = self.robot.forward_kinematics(joint_angles[i])
                if np.any(np.isnan(poses[i])) or np.any(np.isinf(poses[i])):
                    valid_mask[i] = False
            except Exception:
                valid_mask[i] = False

        joint_angles = joint_angles[valid_mask]
        poses = poses[valid_mask]
        print(f"Valid samples: {len(joint_angles)}/{n_samples}")

        np.savez(self.data_dir / "dataset_singularity.npz",
                 poses=poses, joint_angles=joint_angles)
        return poses, joint_angles

    def generate_boundary(self, n_samples=25000):
        """Phase 2c: Samples near joint limits (workspace boundary)."""
        print(f"\n=== Generating {n_samples} boundary samples ===")
        joint_angles = np.zeros((n_samples, 6))
        limits_range = self.robot.joint_limits_upper - self.robot.joint_limits_lower

        for i in range(n_samples):
            q = self.robot.random_joint_config()
            # For 2-3 random joints, push near a limit
            n_boundary_joints = np.random.randint(2, 4)
            boundary_joints = np.random.choice(6, n_boundary_joints, replace=False)
            for j in boundary_joints:
                if np.random.random() > 0.5:
                    # Near upper limit
                    q[j] = self.robot.joint_limits_upper[j] - np.abs(np.random.normal(0, 0.1 * limits_range[j]))
                else:
                    # Near lower limit
                    q[j] = self.robot.joint_limits_lower[j] + np.abs(np.random.normal(0, 0.1 * limits_range[j]))
            q = np.clip(q, self.robot.joint_limits_lower, self.robot.joint_limits_upper)
            joint_angles[i] = q

        poses = np.zeros((n_samples, 6))
        valid_mask = np.ones(n_samples, dtype=bool)
        for i in tqdm(range(n_samples), desc="FK (boundary)"):
            try:
                poses[i] = self.robot.forward_kinematics(joint_angles[i])
                if np.any(np.isnan(poses[i])) or np.any(np.isinf(poses[i])):
                    valid_mask[i] = False
            except Exception:
                valid_mask[i] = False

        joint_angles = joint_angles[valid_mask]
        poses = poses[valid_mask]
        print(f"Valid samples: {len(joint_angles)}/{n_samples}")

        np.savez(self.data_dir / "dataset_boundary.npz",
                 poses=poses, joint_angles=joint_angles)
        return poses, joint_angles

    def combine_and_preprocess(self):
        """Combine all datasets, normalize, and split."""
        print("\n=== Combining and preprocessing datasets ===")

        all_poses = []
        all_joints = []
        for name in ["dataset_uniform.npz", "dataset_singularity.npz", "dataset_boundary.npz"]:
            path = self.data_dir / name
            if path.exists():
                data = np.load(path)
                all_poses.append(data['poses'])
                all_joints.append(data['joint_angles'])
                print(f"  Loaded {name}: {len(data['poses'])} samples")

        poses = np.concatenate(all_poses, axis=0)
        joints = np.concatenate(all_joints, axis=0)
        print(f"  Total combined: {len(poses)} samples")

        # Shuffle
        indices = np.random.permutation(len(poses))
        poses = poses[indices]
        joints = joints[indices]

        # Normalize
        normalizer = Normalizer()
        normalizer.fit(poses, joints)
        normalizer.save(str(self.data_dir / "normalization_params.npz"))

        # Split: 70/15/15
        n = len(poses)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        splits = {
            'train': (poses[:n_train], joints[:n_train]),
            'val': (poses[n_train:n_train + n_val], joints[n_train:n_train + n_val]),
            'test': (poses[n_train + n_val:], joints[n_train + n_val:]),
        }

        for split_name, (p, j) in splits.items():
            np.savez(self.data_dir / f"{split_name}.npz", poses=p, joint_angles=j)
            print(f"  {split_name}: {len(p)} samples")

        # Save combined
        np.savez(self.data_dir / "dataset_combined.npz", poses=poses, joint_angles=joints)

        # Log stats
        stats = {
            "total_samples": int(n),
            "train_samples": int(n_train),
            "val_samples": int(n_val),
            "test_samples": int(n - n_train - n_val),
            "input_stats": {
                "mean": normalizer.input_mean.tolist(),
                "std": normalizer.input_std.tolist(),
                "min": poses.min(axis=0).tolist(),
                "max": poses.max(axis=0).tolist()
            },
            "output_stats": {
                "mean": normalizer.output_mean.tolist(),
                "std": normalizer.output_std.tolist(),
                "min": joints.min(axis=0).tolist(),
                "max": joints.max(axis=0).tolist()
            }
        }
        log_dataset_stats(stats)
        print("  ✅ Dataset preprocessing complete!")
        return stats


def generate_all_data():
    """Run full data generation pipeline."""
    gen = DataGenerator()
    gen.generate_uniform(n_samples=100000)
    gen.generate_singularity(n_samples=50000)
    gen.generate_boundary(n_samples=25000)
    stats = gen.combine_and_preprocess()
    update_phase("phase2", "Data generation complete")
    return stats


if __name__ == "__main__":
    generate_all_data()
