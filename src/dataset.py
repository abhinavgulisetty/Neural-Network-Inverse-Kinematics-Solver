"""
PyTorch Dataset and DataLoader for IK training data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class IKDataset(Dataset):
    """PyTorch Dataset for inverse kinematics training."""

    def __init__(self, data_path, normalize=True, norm_params_path=None):
        """
        Args:
            data_path: path to .npz file with 'poses' and 'joint_angles'
            normalize: whether to normalize inputs/outputs
            norm_params_path: path to normalization_params.npz
        """
        data = np.load(data_path)
        self.poses = data['poses'].astype(np.float32)
        self.joint_angles = data['joint_angles'].astype(np.float32)

        if normalize and norm_params_path is not None:
            params = np.load(norm_params_path)
            input_mean = params['input_mean'].astype(np.float32)
            input_std = params['input_std'].astype(np.float32)
            output_mean = params['output_mean'].astype(np.float32)
            output_std = params['output_std'].astype(np.float32)

            self.poses = (self.poses - input_mean) / input_std
            self.joint_angles = (self.joint_angles - output_mean) / output_std

        self.poses = torch.from_numpy(self.poses)
        self.joint_angles = torch.from_numpy(self.joint_angles)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx], self.joint_angles[idx]


def create_dataloaders(data_dir, batch_size=256, normalize=True, num_workers=0):
    """
    Create train/val/test DataLoaders.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    data_dir = Path(data_dir)
    norm_path = data_dir / "normalization_params.npz" if normalize else None

    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = IKDataset(
            data_dir / f"{split}.npz",
            normalize=normalize,
            norm_params_path=norm_path
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=False  # CPU training
        )
        print(f"  {split}: {len(dataset)} samples, {len(loaders[split])} batches")

    return loaders
