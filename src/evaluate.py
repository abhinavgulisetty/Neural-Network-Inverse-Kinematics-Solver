"""
Comprehensive evaluation metrics for the IK solver.
"""
import time
import json
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_model import RobotModel
from src.utils import Normalizer, ensure_dir


def evaluate_model(model, iteration, data_dir, results_dir,
                   is_sincos=False, device='cpu'):
    """
    Full evaluation of a trained model on test data.

    Returns:
        metrics dict
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    test_data = np.load(data_dir / "test.npz")
    test_poses = test_data['poses'].astype(np.float32)
    test_joints = test_data['joint_angles'].astype(np.float32)

    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))

    test_poses_norm = normalizer.normalize_input(test_poses)
    test_joints_norm = normalizer.normalize_output(test_joints)

    robot = RobotModel()
    model.eval()

    n_test = len(test_poses)
    print(f"\n  Evaluating on {n_test} test samples...")

    with torch.no_grad():
        x = torch.from_numpy(test_poses_norm.astype(np.float32)).to(device)

        if is_sincos:
            pred_sincos = model(x)
            pred_norm = torch.zeros(n_test, 6, device=device)
            for i in range(6):
                sin_v = pred_sincos[:, 2*i]
                cos_v = pred_sincos[:, 2*i + 1]
                pred_norm[:, i] = torch.atan2(sin_v, cos_v)
            pred_joints_norm = pred_norm.numpy()
            pred_joints = normalizer.denormalize_output(pred_joints_norm)
        else:
            pred_norm = model(x).numpy()
            pred_joints = normalizer.denormalize_output(pred_norm)

    position_errors_mm = []
    orientation_errors_deg = []

    n_eval = min(n_test, 5000)
    for i in range(n_eval):
        try:
            achieved_pose = robot.forward_kinematics(pred_joints[i])
            target_pose = test_poses[i]

            pos_err = np.linalg.norm(achieved_pose[:3] - target_pose[:3]) * 1000
            position_errors_mm.append(pos_err)

            ori_err = np.linalg.norm(achieved_pose[3:] - target_pose[3:])
            ori_err_deg = np.degrees(ori_err)
            orientation_errors_deg.append(ori_err_deg)
        except Exception:
            position_errors_mm.append(float('inf'))
            orientation_errors_deg.append(float('inf'))

    pos_errors = np.array(position_errors_mm)
    ori_errors = np.array(orientation_errors_deg)

    valid = np.isfinite(pos_errors) & np.isfinite(ori_errors)
    pos_errors_valid = pos_errors[valid]
    ori_errors_valid = ori_errors[valid]

    inference_times = []
    test_input = torch.from_numpy(test_poses_norm[:1].astype(np.float32)).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)

    for i in range(1000):
        inp = torch.from_numpy(test_poses_norm[i:i+1].astype(np.float32)).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        inference_times.append((time.perf_counter() - start) * 1000)

    inference_times = np.array(inference_times)

    joint_errors_deg = np.degrees(np.abs(pred_joints[:n_eval] - test_joints[:n_eval]))
    joint_rmse_deg = np.sqrt(np.mean(joint_errors_deg**2, axis=0))

    success = (pos_errors_valid < 1.0) & (ori_errors_valid < 0.5)
    success_rate = np.mean(success) * 100 if len(success) > 0 else 0.0

    metrics = {
        "position_rmse_mm": float(np.sqrt(np.mean(pos_errors_valid**2))) if len(pos_errors_valid) > 0 else float('inf'),
        "position_mean_mm": float(np.mean(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_median_mm": float(np.median(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_95th_mm": float(np.percentile(pos_errors_valid, 95)) if len(pos_errors_valid) > 0 else float('inf'),
        "position_max_mm": float(np.max(pos_errors_valid)) if len(pos_errors_valid) > 0 else float('inf'),
        "orientation_rmse_deg": float(np.sqrt(np.mean(ori_errors_valid**2))) if len(ori_errors_valid) > 0 else float('inf'),
        "orientation_mean_deg": float(np.mean(ori_errors_valid)) if len(ori_errors_valid) > 0 else float('inf'),
        "orientation_median_deg": float(np.median(ori_errors_valid)) if len(ori_errors_valid) > 0 else float('inf'),
        "orientation_95th_deg": float(np.percentile(ori_errors_valid, 95)) if len(ori_errors_valid) > 0 else float('inf'),
        "success_rate_pct": float(success_rate),
        "avg_inference_ms": float(np.mean(inference_times)),
        "max_inference_ms": float(np.max(inference_times)),
        "std_inference_ms": float(np.std(inference_times)),
        "joint_rmse_deg": joint_rmse_deg.tolist(),
        "n_evaluated": int(n_eval),
        "n_valid": int(np.sum(valid)),
    }

    print(f"\n  === Iteration {iteration} Evaluation Results ===")
    print(f"  Position RMSE:     {metrics['position_rmse_mm']:.4f} mm")
    print(f"  Position 95th:     {metrics['position_95th_mm']:.4f} mm")
    print(f"  Orientation RMSE:  {metrics['orientation_rmse_deg']:.4f} deg")
    print(f"  Success Rate:      {metrics['success_rate_pct']:.1f}%")
    print(f"  Avg Inference:     {metrics['avg_inference_ms']:.4f} ms")
    print(f"  Joint RMSE (deg):  {[f'{x:.2f}' for x in metrics['joint_rmse_deg']]}")

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


def run_numerical_ik_benchmark(data_dir, results_dir, n_samples=500):
    """Benchmark the numerical IK solver for comparison."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    test_data = np.load(data_dir / "test.npz")
    test_poses = test_data['poses']

    robot = RobotModel()

    print(f"\n  Running numerical IK on {n_samples} samples...")
    solve_times = []
    successes = 0
    pos_errors = []

    for i in range(min(n_samples, len(test_poses))):
        q_sol, success, solve_time = robot.numerical_ik(test_poses[i])
        solve_times.append(solve_time)
        if success:
            successes += 1
            achieved = robot.forward_kinematics(q_sol)
            err = np.linalg.norm(achieved[:3] - test_poses[i][:3]) * 1000
            pos_errors.append(err)

    numerical_metrics = {
        "avg_solve_time_ms": float(np.mean(solve_times)),
        "max_solve_time_ms": float(np.max(solve_times)),
        "success_rate_pct": float(successes / n_samples * 100),
        "position_rmse_mm": float(np.sqrt(np.mean(np.array(pos_errors)**2))) if pos_errors else float('inf'),
        "n_samples": n_samples
    }

    with open(results_dir / "numerical_ik_benchmark.json", 'w') as f:
        json.dump(numerical_metrics, f, indent=2)

    print(f"  Numerical IK:")
    print(f"    Avg solve time: {numerical_metrics['avg_solve_time_ms']:.2f} ms")
    print(f"    Success rate: {numerical_metrics['success_rate_pct']:.1f}%")
    print(f"    Position RMSE: {numerical_metrics['position_rmse_mm']:.4f} mm")

    return numerical_metrics


def compile_all_metrics(results_dir):
    """Compile all iteration metrics and numerical baseline into one file."""
    results_dir = Path(results_dir)
    all_metrics = {"iterations": [], "numerical_baseline": None}

    for i in range(1, 10):
        path = results_dir / f"metrics_iter{i}.json"
        if path.exists():
            with open(path) as f:
                m = json.load(f)
                m['iteration'] = i
                all_metrics['iterations'].append(m)

    num_path = results_dir / "numerical_ik_benchmark.json"
    if num_path.exists():
        with open(num_path) as f:
            all_metrics['numerical_baseline'] = json.load(f)

    if all_metrics['iterations']:
        best = min(all_metrics['iterations'],
                   key=lambda x: x.get('position_rmse_mm', float('inf')))
        all_metrics['best_iteration'] = best['iteration']

        if all_metrics['numerical_baseline']:
            num_time = all_metrics['numerical_baseline']['avg_solve_time_ms']
            nn_time = best['avg_inference_ms']
            all_metrics['speedup_factor'] = num_time / nn_time if nn_time > 0 else 0
    else:
        all_metrics['best_iteration'] = None
        all_metrics['speedup_factor'] = 0

    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    compile_all_metrics(project_root / "results")
