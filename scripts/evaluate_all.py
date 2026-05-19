#!/usr/bin/env python3

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

import torch
import numpy as np
from pathlib import Path

from src.model import create_model
from src.robot_model import RobotModel
from src.utils import Normalizer


def evaluate_model(model_version, model_path, test_x, test_y, normalizer, robot):
    device = "cpu"
    
    model, arch_name = create_model(model_version, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    X_norm = normalizer.normalize_input(test_x).astype(np.float32)
    X_tensor = torch.from_numpy(X_norm).to(device)
    
    n_samples = len(test_x)
    predictions = np.zeros((n_samples, 6))
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = X_tensor[i:i+batch_size]
            
            if model_version in [4, 6, 7]:
                pred_sincos = model(batch_x)
                for j in range(6):
                    sin_val = pred_sincos[:, 2*j]
                    cos_val = pred_sincos[:, 2*j + 1]
                    predictions[i:i+batch_size, j] = torch.atan2(sin_val, cos_val).cpu().numpy()
            elif model_version == 8:
                z = torch.zeros(batch_x.size(0), 6, device=device)
                pred_norm = model.inverse(batch_x, z)
                predictions[i:i+batch_size] = normalizer.denormalize_output(pred_norm.cpu().numpy())
            elif model_version == 9:
                pred = model.sample(batch_x, n_steps=100)
                predictions[i:i+batch_size] = pred.cpu().numpy()
            elif model_version == 10:
                pred = model.predict_angles(batch_x)
                predictions[i:i+batch_size] = pred.cpu().numpy()
            else:
                pred_norm = model(batch_x)
                predictions[i:i+batch_size] = normalizer.denormalize_output(pred_norm.cpu().numpy())
    
    joint_error_rad = np.abs(predictions - test_y)
    joint_error_deg = np.rad2deg(joint_error_rad)
    
    pos_errors = []
    for i in range(min(1000, n_samples)):
        target_pose = test_x[i]
        pred_joints = predictions[i]
        
        pred_pose = robot.forward_kinematics(pred_joints)
        if pred_pose is not None:
            pos_error = np.linalg.norm(pred_pose[:3] - target_pose[:3]) * 1000
            pos_errors.append(pos_error)
    
    pos_error_mean = np.mean(pos_errors) if pos_errors else float('nan')
    pos_error_max = np.max(pos_errors) if pos_errors else float('nan')
    
    return {
        'model_version': model_version,
        'arch_name': arch_name,
        'joint_error_mean_deg': joint_error_deg.mean(),
        'joint_error_max_deg': joint_error_deg.max(),
        'pos_error_mean_mm': pos_error_mean,
        'pos_error_max_mm': pos_error_max,
    }


def main():
    data_dir = Path("data")
    models_dir = Path("models")
    
    print("Loading test data...")
    test_data = np.load(data_dir / "test.npz")
    test_x = test_data['poses'].astype(np.float32)
    test_y = test_data['joint_angles'].astype(np.float32)
    print(f"  Test samples: {len(test_x)}")
    
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    robot = RobotModel()
    
    results = []
    for version in range(1, 11):
        model_path = models_dir / f"best_model_iter{version}.pth"
        if model_path.exists():
            print(f"\nEvaluating V{version}...")
            try:
                result = evaluate_model(version, model_path, test_x, test_y, normalizer, robot)
                results.append(result)
                print(f"  {result['arch_name']}")
                print(f"  Joint Error: {result['joint_error_mean_deg']:.2f}° (mean), {result['joint_error_max_deg']:.2f}° (max)")
                print(f"  Position Error: {result['pos_error_mean_mm']:.2f}mm (mean), {result['pos_error_max_mm']:.2f}mm (max)")
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<6} {'Architecture':<45} {'Joint Err (°)':<15} {'Pos Err (mm)':<15}")
    print("-"*80)
    for r in results:
        print(f"V{r['model_version']:<5} {r['arch_name']:<45} {r['joint_error_mean_deg']:<15.2f} {r['pos_error_mean_mm']:<15.2f}")
    
    if results:
        best = min(results, key=lambda x: x['pos_error_mean_mm'])
        print(f"\nBest model by position error: V{best['model_version']} ({best['pos_error_mean_mm']:.2f}mm)")


if __name__ == "__main__":
    main()
