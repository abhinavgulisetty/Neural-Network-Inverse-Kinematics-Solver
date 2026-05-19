import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_model import RobotModel
from src.utils import Normalizer, ensure_dir
from src.models_advanced import (
    IKNetV6, IKNetV7, IKNetV8, IKNetV9, IKNetV10,
    DifferentiableFKLayer
)
from src.train_advanced import TestTimeOptimizer


def evaluate_advanced_model(
    model,
    iteration: int,
    data_dir: Path,
    results_dir: Path,
    device: str = 'cpu',
    use_test_time_opt: bool = False,
    n_tto_steps: int = 10
) -> Dict:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    
    test_data = np.load(data_dir / "test.npz")
    test_poses = test_data['poses'].astype(np.float32)
    test_joints = test_data['joint_angles'].astype(np.float32)
    
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    test_poses_norm = normalizer.normalize_input(test_poses)
    
    robot = RobotModel()
    fk_layer = DifferentiableFKLayer().to(device)
    
    model.eval()
    model = model.to(device)
    
    n_test = len(test_poses)
    n_eval = min(n_test, 5000)
    
    print(f"\n  Evaluating on {n_eval} test samples...")
    
    is_transformer = isinstance(model, IKNetV6)
    is_pinn = isinstance(model, IKNetV7)
    is_cinn = isinstance(model, IKNetV8)
    is_diffusion = isinstance(model, IKNetV9)
    is_mdn = isinstance(model, IKNetV10)
    
    all_pred_joints = []
    all_uncertainties = []
    inference_times = []
    
    with torch.no_grad():
        for i in range(n_eval):
            pose_norm = torch.from_numpy(test_poses_norm[i:i+1]).to(device)
            pose_denorm = torch.from_numpy(test_poses[i:i+1]).to(device)
            
            start_time = time.perf_counter()
            
            if is_diffusion:
                pred_joints = model.predict_angles(pose_denorm, n_steps=50)
            elif is_cinn:
                pred_joints = model.predict_angles(pose_denorm)
            elif is_mdn:
                pred_joints, uncertainty = model.predict_angles(pose_norm, return_uncertainty=True)
                all_uncertainties.append(uncertainty.item())
            else:
                pred_sincos = model(pose_norm)
                pred_joints = torch.zeros(1, 6, device=device)
                for j in range(6):
                    pred_joints[0, j] = torch.atan2(pred_sincos[0, 2*j], pred_sincos[0, 2*j + 1])
            
            inference_time = (time.perf_counter() - start_time) * 1000
            inference_times.append(inference_time)
            
            all_pred_joints.append(pred_joints.cpu().numpy())
    
    pred_joints = np.concatenate(all_pred_joints, axis=0)
    inference_times = np.array(inference_times)
    
    if use_test_time_opt:
        print("  Running test-time optimization...")
        tto = TestTimeOptimizer(robot, n_steps=n_tto_steps, device=device)
        pred_joints_tto = []
        tto_times = []
        
        for i in range(n_eval):
            joints_init = torch.from_numpy(pred_joints[i:i+1]).float().to(device)
            pose_target = torch.from_numpy(test_poses[i:i+1]).float().to(device)
            
            start_time = time.perf_counter()
            joints_refined = tto.refine(joints_init, pose_target)
            tto_time = (time.perf_counter() - start_time) * 1000
            
            pred_joints_tto.append(joints_refined.cpu().numpy())
            tto_times.append(tto_time)
        
        pred_joints_tto = np.concatenate(pred_joints_tto, axis=0)
        tto_times = np.array(tto_times)
    
    position_errors_mm = []
    orientation_errors_deg = []
    position_errors_tto_mm = []
    orientation_errors_tto_deg = []
    
    for i in range(n_eval):
        try:
            achieved_pose = robot.forward_kinematics(pred_joints[i])
            target_pose = test_poses[i]
            
            pos_err = np.linalg.norm(achieved_pose[:3] - target_pose[:3]) * 1000
            position_errors_mm.append(pos_err)
            
            ori_err = np.linalg.norm(achieved_pose[3:] - target_pose[3:])
            orientation_errors_deg.append(np.degrees(ori_err))
            
            if use_test_time_opt:
                achieved_pose_tto = robot.forward_kinematics(pred_joints_tto[i])
                pos_err_tto = np.linalg.norm(achieved_pose_tto[:3] - target_pose[:3]) * 1000
                position_errors_tto_mm.append(pos_err_tto)
                ori_err_tto = np.linalg.norm(achieved_pose_tto[3:] - target_pose[3:])
                orientation_errors_tto_deg.append(np.degrees(ori_err_tto))
                
        except Exception as e:
            position_errors_mm.append(float('inf'))
            orientation_errors_deg.append(float('inf'))
            if use_test_time_opt:
                position_errors_tto_mm.append(float('inf'))
                orientation_errors_tto_deg.append(float('inf'))
    
    pos_errors = np.array(position_errors_mm)
    ori_errors = np.array(orientation_errors_deg)
    
    valid = np.isfinite(pos_errors) & np.isfinite(ori_errors)
    pos_errors_valid = pos_errors[valid]
    ori_errors_valid = ori_errors[valid]
    
    joint_errors_deg = np.degrees(np.abs(pred_joints[:n_eval] - test_joints[:n_eval]))
    joint_rmse_deg = np.sqrt(np.mean(joint_errors_deg**2, axis=0))
    
    success = (pos_errors_valid < 1.0) & (ori_errors_valid < 0.5)
    success_rate = np.mean(success) * 100 if len(success) > 0 else 0.0
    
    success_5mm = (pos_errors_valid < 5.0) & (ori_errors_valid < 2.0)
    success_rate_5mm = np.mean(success_5mm) * 100 if len(success_5mm) > 0 else 0.0
    
    metrics = {
        "iteration": iteration,
        "model_type": type(model).__name__,
        
        "position_rmse_mm": float(np.sqrt(np.mean(pos_errors_valid**2))) if len(pos_errors_valid) > 0 else float('inf'),
        "position_mean_mm": float(np.mean(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_median_mm": float(np.median(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_std_mm": float(np.std(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_95th_mm": float(np.percentile(pos_errors_valid, 95)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_99th_mm": float(np.percentile(pos_errors_valid, 99)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_max_mm": float(np.max(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        
        "orientation_rmse_deg": float(np.sqrt(np.mean(ori_errors_valid**2))) if len(ori_errors_valid) > 0 else float('inf'),
        "orientation_mean_deg": float(np.mean(ori_errors_valid)) if len(ori_errors_valid) > 0 else float('inf'),
        "orientation_median_deg": float(np.median(ori_errors_valid)) if len(ori_errors_valid) > 0 else float('inf'),
        "orientation_95th_deg": float(np.percentile(ori_errors_valid, 95)) if len(ori_errors_valid) > 0 else float('inf'),
        
        "success_rate_1mm_pct": float(success_rate),
        "success_rate_5mm_pct": float(success_rate_5mm),
        
        "avg_inference_ms": float(np.mean(inference_times)),
        "median_inference_ms": float(np.median(inference_times)),
        "max_inference_ms": float(np.max(inference_times)),
        "std_inference_ms": float(np.std(inference_times)),
        
        "joint_rmse_deg": joint_rmse_deg.tolist(),
        "joint_mean_error_deg": np.mean(joint_errors_deg, axis=0).tolist(),
        "joint_max_error_deg": np.max(joint_errors_deg, axis=0).tolist(),
        
        "n_evaluated": int(n_eval),
        "n_valid": int(np.sum(valid)),
    }
    
    if use_test_time_opt:
        pos_errors_tto = np.array(position_errors_tto_mm)
        ori_errors_tto = np.array(orientation_errors_tto_deg)
        valid_tto = np.isfinite(pos_errors_tto) & np.isfinite(ori_errors_tto)
        
        metrics.update({
            "tto_position_rmse_mm": float(np.sqrt(np.mean(pos_errors_tto[valid_tto]**2))),
            "tto_position_mean_mm": float(np.mean(pos_errors_tto[valid_tto])),
            "tto_orientation_rmse_deg": float(np.sqrt(np.mean(ori_errors_tto[valid_tto]**2))),
            "tto_avg_time_ms": float(np.mean(tto_times)),
            "tto_improvement_pct": float((metrics['position_rmse_mm'] - np.sqrt(np.mean(pos_errors_tto[valid_tto]**2))) / metrics['position_rmse_mm'] * 100)
        })
    
    if is_mdn and len(all_uncertainties) > 0:
        uncertainties = np.array(all_uncertainties)
        metrics.update({
            "uncertainty_mean": float(np.mean(uncertainties)),
            "uncertainty_std": float(np.std(uncertainties)),
            "uncertainty_correlation_with_error": float(np.corrcoef(uncertainties, pos_errors[:len(uncertainties)])[0, 1])
        })
    
    print(f"\n  === Iteration {iteration} ({type(model).__name__}) Results ===")
    print(f"  Position RMSE:     {metrics['position_rmse_mm']:.4f} mm")
    print(f"  Position Median:   {metrics['position_median_mm']:.4f} mm")
    print(f"  Position 95th:     {metrics['position_95th_mm']:.4f} mm")
    print(f"  Orientation RMSE:  {metrics['orientation_rmse_deg']:.4f} deg")
    print(f"  Success Rate (1mm): {metrics['success_rate_1mm_pct']:.1f}%")
    print(f"  Success Rate (5mm): {metrics['success_rate_5mm_pct']:.1f}%")
    print(f"  Avg Inference:     {metrics['avg_inference_ms']:.3f} ms")
    print(f"  Joint RMSE (deg):  {[f'{x:.2f}' for x in metrics['joint_rmse_deg']]}")
    
    if use_test_time_opt:
        print(f"  TTO Position RMSE: {metrics['tto_position_rmse_mm']:.4f} mm")
        print(f"  TTO Improvement:   {metrics['tto_improvement_pct']:.1f}%")
    
    np.savez(results_dir / f"errors_iter{iteration}.npz",
             position_errors_mm=pos_errors,
             orientation_errors_deg=ori_errors,
             pred_joints=pred_joints[:n_eval],
             true_joints=test_joints[:n_eval],
             inference_times=inference_times)
    
    metrics_path = results_dir / f"metrics_iter{iteration}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def evaluate_multi_solution(
    model,
    data_dir: Path,
    n_samples_per_pose: int = 10,
    n_poses: int = 500,
    device: str = 'cpu'
) -> Dict:
    data_dir = Path(data_dir)
    
    test_data = np.load(data_dir / "test.npz")
    test_poses = test_data['poses'][:n_poses].astype(np.float32)
    
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    test_poses_norm = normalizer.normalize_input(test_poses)
    
    robot = RobotModel()
    fk_layer = DifferentiableFKLayer().to(device)
    
    model.eval()
    
    is_cinn = isinstance(model, IKNetV8)
    is_diffusion = isinstance(model, IKNetV9)
    is_mdn = isinstance(model, IKNetV10)
    
    if not (is_cinn or is_diffusion or is_mdn):
        print("  Model does not support multi-solution sampling")
        return {}
    
    print(f"\n  Evaluating multi-solution capability ({n_samples_per_pose} samples per pose)...")
    
    all_diversities = []
    all_valid_counts = []
    all_best_errors = []
    
    with torch.no_grad():
        for i in range(n_poses):
            pose_denorm = torch.from_numpy(test_poses[i:i+1]).to(device)
            pose_norm = torch.from_numpy(test_poses_norm[i:i+1]).to(device)
            
            if is_cinn:
                samples = model.sample(pose_denorm, n_samples=n_samples_per_pose, temperature=1.0)
            elif is_diffusion:
                samples = []
                for _ in range(n_samples_per_pose):
                    sample = model.sample(pose_denorm, n_steps=50)
                    samples.append(sample)
                samples = torch.stack(samples, dim=1)
            elif is_mdn:
                samples = model.sample(pose_norm, n_samples=n_samples_per_pose)
            
            samples = samples.squeeze(0)
            
            pos_errors = []
            for j in range(n_samples_per_pose):
                try:
                    achieved = robot.forward_kinematics(samples[j].cpu().numpy())
                    pos_err = np.linalg.norm(achieved[:3] - test_poses[i, :3]) * 1000
                    pos_errors.append(pos_err)
                except:
                    pos_errors.append(float('inf'))
            
            pos_errors = np.array(pos_errors)
            valid_mask = pos_errors < 50
            
            valid_samples = samples[valid_mask].cpu().numpy() if valid_mask.sum() > 1 else np.array([])
            if len(valid_samples) > 1:
                diversity = np.mean([
                    np.linalg.norm(valid_samples[a] - valid_samples[b])
                    for a in range(len(valid_samples))
                    for b in range(a+1, len(valid_samples))
                ])
            else:
                diversity = 0.0
            
            all_diversities.append(diversity)
            all_valid_counts.append(valid_mask.sum())
            all_best_errors.append(np.min(pos_errors[np.isfinite(pos_errors)]) if np.any(np.isfinite(pos_errors)) else float('inf'))
    
    metrics = {
        "multi_solution": {
            "avg_diversity_rad": float(np.mean(all_diversities)),
            "avg_valid_samples": float(np.mean(all_valid_counts)),
            "avg_best_error_mm": float(np.mean([e for e in all_best_errors if np.isfinite(e)])),
            "pct_with_multiple_solutions": float(np.mean([c > 1 for c in all_valid_counts]) * 100)
        }
    }
    
    print(f"  Avg diversity: {metrics['multi_solution']['avg_diversity_rad']:.4f} rad")
    print(f"  Avg valid samples: {metrics['multi_solution']['avg_valid_samples']:.1f}/{n_samples_per_pose}")
    print(f"  Avg best error: {metrics['multi_solution']['avg_best_error_mm']:.2f} mm")
    print(f"  % with multiple solutions: {metrics['multi_solution']['pct_with_multiple_solutions']:.1f}%")
    
    return metrics


def evaluate_workspace_regions(
    model,
    data_dir: Path,
    results_dir: Path,
    device: str = 'cpu'
) -> Dict:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    
    robot = RobotModel()
    
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    regions = {}
    for region_name in ['uniform', 'singularity', 'boundary']:
        path = data_dir / f"dataset_{region_name}.npz"
        if path.exists():
            data = np.load(path)
            regions[region_name] = {
                'poses': data['poses'][:1000],
                'joints': data['joint_angles'][:1000]
            }
    
    model.eval()
    
    is_mdn = isinstance(model, IKNetV10)
    is_cinn = isinstance(model, IKNetV8)
    is_diffusion = isinstance(model, IKNetV9)
    
    region_metrics = {}
    
    for region_name, region_data in regions.items():
        poses = region_data['poses'].astype(np.float32)
        joints = region_data['joints'].astype(np.float32)
        poses_norm = normalizer.normalize_input(poses)
        
        pos_errors = []
        
        with torch.no_grad():
            for i in range(len(poses)):
                pose_norm = torch.from_numpy(poses_norm[i:i+1]).to(device)
                pose_denorm = torch.from_numpy(poses[i:i+1]).to(device)
                
                if is_diffusion:
                    pred = model.predict_angles(pose_denorm, n_steps=50)
                elif is_cinn:
                    pred = model.predict_angles(pose_denorm)
                elif is_mdn:
                    pred = model.predict_angles(pose_norm)
                else:
                    pred_sincos = model(pose_norm)
                    pred = torch.zeros(1, 6, device=device)
                    for j in range(6):
                        pred[0, j] = torch.atan2(pred_sincos[0, 2*j], pred_sincos[0, 2*j + 1])
                
                try:
                    achieved = robot.forward_kinematics(pred[0].cpu().numpy())
                    pos_err = np.linalg.norm(achieved[:3] - poses[i, :3]) * 1000
                    pos_errors.append(pos_err)
                except:
                    pos_errors.append(float('inf'))
        
        pos_errors = np.array(pos_errors)
        valid = np.isfinite(pos_errors)
        
        region_metrics[region_name] = {
            "position_rmse_mm": float(np.sqrt(np.mean(pos_errors[valid]**2))) if valid.sum() > 0 else float('inf'),
            "position_median_mm": float(np.median(pos_errors[valid])) if valid.sum() > 0 else float('inf'),
            "position_95th_mm": float(np.percentile(pos_errors[valid], 95)) if valid.sum() > 0 else float('inf'),
            "success_rate_5mm_pct": float(np.mean(pos_errors[valid] < 5.0) * 100) if valid.sum() > 0 else 0.0,
            "n_samples": len(poses)
        }
        
        print(f"\n  Region: {region_name}")
        print(f"    Position RMSE: {region_metrics[region_name]['position_rmse_mm']:.2f} mm")
        print(f"    Position 95th: {region_metrics[region_name]['position_95th_mm']:.2f} mm")
        print(f"    Success (5mm): {region_metrics[region_name]['success_rate_5mm_pct']:.1f}%")
    
    with open(results_dir / "region_analysis.json", 'w') as f:
        json.dump(region_metrics, f, indent=2)
    
    return region_metrics


def run_comprehensive_benchmark(
    iterations: List[int] = [6, 7, 8, 9, 10],
    use_tto: bool = True
):
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    results_dir = project_root / "results"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_metrics = {"iterations": [], "comparison": {}}
    
    for iteration in iterations:
        model_path = model_dir / f"best_model_iter{iteration}.pth"
        if not model_path.exists():
            print(f"\n  Model for iteration {iteration} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Benchmarking Iteration {iteration}")
        print(f"{'='*60}")
        
        from src.models_advanced import create_advanced_model
        model, _ = create_advanced_model(iteration, device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        metrics = evaluate_advanced_model(
            model, iteration, data_dir, results_dir,
            device=device, use_test_time_opt=use_tto, n_tto_steps=10
        )
        
        if iteration in [8, 9, 10]:
            multi_metrics = evaluate_multi_solution(
                model, data_dir, n_samples_per_pose=10, n_poses=500, device=device
            )
            metrics.update(multi_metrics)
        
        region_metrics = evaluate_workspace_regions(
            model, data_dir, results_dir, device=device
        )
        metrics['region_analysis'] = region_metrics
        
        all_metrics['iterations'].append(metrics)
    
    if len(all_metrics['iterations']) > 0:
        best_pos = min(all_metrics['iterations'], key=lambda x: x.get('position_rmse_mm', float('inf')))
        best_success = max(all_metrics['iterations'], key=lambda x: x.get('success_rate_5mm_pct', 0))
        fastest = min(all_metrics['iterations'], key=lambda x: x.get('avg_inference_ms', float('inf')))
        
        all_metrics['comparison'] = {
            "best_position_accuracy": {
                "iteration": best_pos['iteration'],
                "position_rmse_mm": best_pos['position_rmse_mm']
            },
            "best_success_rate": {
                "iteration": best_success['iteration'],
                "success_rate_5mm_pct": best_success['success_rate_5mm_pct']
            },
            "fastest_inference": {
                "iteration": fastest['iteration'],
                "avg_inference_ms": fastest['avg_inference_ms']
            }
        }
        
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Best Position Accuracy: Iter {best_pos['iteration']} ({best_pos['position_rmse_mm']:.2f} mm)")
        print(f"Best Success Rate (5mm): Iter {best_success['iteration']} ({best_success['success_rate_5mm_pct']:.1f}%)")
        print(f"Fastest Inference: Iter {fastest['iteration']} ({fastest['avg_inference_ms']:.3f} ms)")
    
    with open(results_dir / "comprehensive_benchmark.json", 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, nargs='+', default=[6, 7, 8, 9, 10])
    parser.add_argument('--no_tto', action='store_true', help='Disable test-time optimization')
    args = parser.parse_args()
    
    run_comprehensive_benchmark(
        iterations=args.iterations,
        use_tto=not args.no_tto
    )
