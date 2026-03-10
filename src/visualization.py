"""
Visualization: plots, 3D arm rendering, animated GIFs.
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_model import RobotModel
from src.utils import ensure_dir


def plot_training_curves(results_dir, output_dir):
    """Plot training loss curves for all iterations."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0a0f')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#12121a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    colors = ['#00d4ff', '#8b5cf6', '#f97316', '#22c55e', '#ef4444']

    for i in range(1, 6):
        hist_path = results_dir / f"training_history_iter{i}.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        color = colors[(i-1) % len(colors)]
        ax1.plot(hist['train_loss'], color=color, alpha=0.8, label=f'Iter {i} Train')
        ax2.plot(hist['val_loss'], color=color, alpha=0.8, label=f'Iter {i} Val')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax1.set_yscale('log')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Validation Loss')
    ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss_curves.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0f')
    plt.close()


def plot_error_histograms(results_dir, output_dir):
    """Plot position and orientation error distributions."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Find best iteration
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_m = json.load(f)
        best_iter = all_m.get('best_iteration', 1)
    else:
        best_iter = 1

    errors_path = results_dir / f"errors_iter{best_iter}.npz"
    if not errors_path.exists():
        print(f"  No error data found for iteration {best_iter}")
        return

    data = np.load(errors_path)
    pos_errors = data['position_errors_mm']
    ori_errors = data['orientation_errors_deg']

    # Filter valid
    pos_valid = pos_errors[np.isfinite(pos_errors)]
    ori_valid = ori_errors[np.isfinite(ori_errors)]

    # Position error histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#12121a')
    ax.hist(pos_valid, bins=100, color='#00d4ff', alpha=0.75, edgecolor='none')
    ax.axvline(1.0, color='#ef4444', linestyle='--', linewidth=2, label='Target: 1 mm')
    ax.axvline(np.mean(pos_valid), color='#22c55e', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(pos_valid):.2f} mm')
    ax.set_xlabel('Position Error (mm)', color='white')
    ax.set_ylabel('Count', color='white')
    ax.set_title(f'Position Error Distribution (Iter {best_iter})', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    plt.tight_layout()
    plt.savefig(output_dir / 'position_error_histogram.png', dpi=150,
                facecolor='#0a0a0f', bbox_inches='tight')
    plt.close()

    # Orientation error histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#12121a')
    ax.hist(ori_valid, bins=100, color='#8b5cf6', alpha=0.75, edgecolor='none')
    ax.axvline(0.5, color='#ef4444', linestyle='--', linewidth=2, label='Target: 0.5°')
    ax.axvline(np.mean(ori_valid), color='#22c55e', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(ori_valid):.2f}°')
    ax.set_xlabel('Orientation Error (degrees)', color='white')
    ax.set_ylabel('Count', color='white')
    ax.set_title(f'Orientation Error Distribution (Iter {best_iter})', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    plt.tight_layout()
    plt.savefig(output_dir / 'orientation_error_histogram.png', dpi=150,
                facecolor='#0a0a0f', bbox_inches='tight')
    plt.close()


def plot_per_joint_error(results_dir, output_dir):
    """Box plot of error per joint."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return
    with open(metrics_path) as f:
        all_m = json.load(f)
    best_iter = all_m.get('best_iteration', 1)

    errors_path = results_dir / f"errors_iter{best_iter}.npz"
    if not errors_path.exists():
        return

    data = np.load(errors_path)
    pred = data['pred_joints']
    true = data['true_joints']
    joint_errors = np.degrees(np.abs(pred - true))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#12121a')

    bp = ax.boxplot([joint_errors[:, i] for i in range(6)],
                    labels=[f'Joint {i+1}' for i in range(6)],
                    patch_artist=True)

    colors = ['#00d4ff', '#8b5cf6', '#f97316', '#22c55e', '#ef4444', '#eab308']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for elem in ['whiskers', 'caps', 'medians']:
        for line in bp[elem]:
            line.set_color('white')
    for flier in bp['fliers']:
        flier.set_markerfacecolor('#999')
        flier.set_markersize(2)

    ax.set_ylabel('Error (degrees)', color='white')
    ax.set_title('Per-Joint Prediction Error', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_joint_error.png', dpi=150,
                facecolor='#0a0a0f', bbox_inches='tight')
    plt.close()


def plot_iteration_comparison(results_dir, output_dir):
    """Bar chart comparing metrics across iterations."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return
    with open(metrics_path) as f:
        all_m = json.load(f)

    iters = all_m.get('iterations', [])
    if not iters:
        return

    iter_nums = [it['iteration'] for it in iters]
    pos_rmse = [it['position_rmse_mm'] for it in iters]
    ori_rmse = [it['orientation_rmse_deg'] for it in iters]
    success = [it['success_rate_pct'] for it in iters]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0a0a0f')

    colors = ['#00d4ff', '#8b5cf6', '#f97316', '#22c55e', '#ef4444']

    for ax in axes:
        ax.set_facecolor('#12121a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    bar_colors = [colors[i % len(colors)] for i in range(len(iter_nums))]

    axes[0].bar([f'Iter {n}' for n in iter_nums], pos_rmse, color=bar_colors, alpha=0.8)
    axes[0].axhline(1.0, color='#ef4444', linestyle='--', label='Target')
    axes[0].set_title('Position RMSE (mm)')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    axes[1].bar([f'Iter {n}' for n in iter_nums], ori_rmse, color=bar_colors, alpha=0.8)
    axes[1].axhline(0.5, color='#ef4444', linestyle='--', label='Target')
    axes[1].set_title('Orientation RMSE (°)')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    axes[2].bar([f'Iter {n}' for n in iter_nums], success, color=bar_colors, alpha=0.8)
    axes[2].axhline(95, color='#22c55e', linestyle='--', label='Target')
    axes[2].set_title('Success Rate (%)')
    axes[2].legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    plt.tight_layout()
    plt.savefig(output_dir / 'iteration_comparison.png', dpi=150,
                facecolor='#0a0a0f', bbox_inches='tight')
    plt.close()


def plot_inference_comparison(results_dir, output_dir):
    """Bar chart: NN vs numerical solver timing."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return
    with open(metrics_path) as f:
        all_m = json.load(f)

    best_iter = all_m.get('best_iteration')
    baseline = all_m.get('numerical_baseline')
    if not best_iter or not baseline:
        return

    best_metrics = None
    for it in all_m['iterations']:
        if it['iteration'] == best_iter:
            best_metrics = it
            break
    if not best_metrics:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#12121a')

    methods = ['Numerical\n(Lev-Marq)', f'Neural Net\n(Iter {best_iter})']
    times = [baseline['avg_solve_time_ms'], best_metrics['avg_inference_ms']]

    bars = ax.bar(methods, times, color=['#ef4444', '#00d4ff'], alpha=0.8, width=0.5)
    ax.set_ylabel('Time (ms)', color='white')
    ax.set_title('Inference Time Comparison', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')

    # Add value labels
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.3f} ms', ha='center', color='white', fontweight='bold')

    speedup = times[0] / times[1] if times[1] > 0 else 0
    ax.text(0.5, 0.85, f'Speedup: {speedup:.0f}×',
            transform=ax.transAxes, ha='center', color='#22c55e',
            fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=150,
                facecolor='#0a0a0f', bbox_inches='tight')
    plt.close()


def plot_arm_3d(joint_angles, title="Robot Arm", save_path=None):
    """Plot a single 3D arm configuration."""
    robot = RobotModel()
    link_pos = robot.get_link_positions(joint_angles)

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0f')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#12121a')

    # Plot links
    ax.plot(link_pos[:, 0], link_pos[:, 1], link_pos[:, 2],
            'o-', color='#00d4ff', linewidth=3, markersize=8, markerfacecolor='#8b5cf6')

    # End effector
    ax.scatter(*link_pos[-1], color='#ef4444', s=100, zorder=5, label='End Effector')

    # Base
    ax.scatter(*link_pos[0], color='#22c55e', s=100, zorder=5, marker='^', label='Base')

    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.set_zlabel('Z (m)', color='white')
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
        plt.close()
    return fig


def generate_arm_animation(joint_trajectory, save_path, title="Arm Trajectory"):
    """
    Generate animated GIF of arm following a trajectory.

    Args:
        joint_trajectory: (N, 6) array of joint angles per frame
        save_path: output path for the GIF
        title: plot title
    """
    robot = RobotModel()
    n_frames = len(joint_trajectory)

    # Precompute all link positions
    all_positions = []
    for q in joint_trajectory:
        try:
            all_positions.append(robot.get_link_positions(q))
        except Exception:
            all_positions.append(all_positions[-1] if all_positions else np.zeros((7, 3)))

    # Compute axis limits
    all_pos_arr = np.array(all_positions)
    margin = 0.1
    xlim = [all_pos_arr[:, :, 0].min() - margin, all_pos_arr[:, :, 0].max() + margin]
    ylim = [all_pos_arr[:, :, 1].min() - margin, all_pos_arr[:, :, 1].max() + margin]
    zlim = [all_pos_arr[:, :, 2].min() - margin, all_pos_arr[:, :, 2].max() + margin]

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0f')
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_facecolor('#12121a')
        pos = all_positions[frame]

        # Draw links
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                'o-', color='#00d4ff', linewidth=3, markersize=8, markerfacecolor='#8b5cf6')

        # End effector trace (up to current frame)
        ee_trace = np.array([all_positions[i][-1] for i in range(frame + 1)])
        ax.plot(ee_trace[:, 0], ee_trace[:, 1], ee_trace[:, 2],
                '-', color='#f97316', alpha=0.5, linewidth=1)

        ax.scatter(*pos[-1], color='#ef4444', s=100, zorder=5)
        ax.scatter(*pos[0], color='#22c55e', s=100, marker='^', zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_title(f'{title} (frame {frame+1}/{n_frames})', color='white')
        ax.tick_params(colors='white')

    # Use every Nth frame to keep GIF manageable
    step = max(1, n_frames // 50)
    frames = list(range(0, n_frames, step))

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(str(save_path), writer='pillow', fps=10)
    plt.close()
    print(f"  Saved animation: {save_path}")


def generate_all_visualizations(results_dir=None, output_dir=None):
    """Generate all visualization plots."""
    project_root = Path(__file__).parent.parent
    if results_dir is None:
        results_dir = project_root / "results"
    if output_dir is None:
        output_dir = project_root / "web" / "static" / "plots"
    ensure_dir(output_dir)

    print("\n=== Generating Visualizations ===")

    print("  Training curves...")
    plot_training_curves(results_dir, output_dir)

    print("  Error histograms...")
    plot_error_histograms(results_dir, output_dir)

    print("  Per-joint errors...")
    plot_per_joint_error(results_dir, output_dir)

    print("  Iteration comparison...")
    plot_iteration_comparison(results_dir, output_dir)

    print("  Inference comparison...")
    plot_inference_comparison(results_dir, output_dir)

    print("  ✅ All visualizations generated!")


if __name__ == "__main__":
    generate_all_visualizations()
