"""
Flask web application for the Neural IK Solver dashboard.
"""
import sys
import json
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, request

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)

RESULTS_DIR = project_root / "results"
DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models"

# Lazy-load the solver
_solver = None


def get_solver():
    global _solver
    if _solver is None:
        try:
            from src.ik_solver import IKSolver
            _solver = IKSolver()
        except Exception as e:
            print(f"Warning: Could not load IK solver: {e}")
    return _solver


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/metrics')
def api_metrics():
    """Return all compiled metrics."""
    path = RESULTS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No metrics available yet"}), 404


@app.route('/api/predict')
def api_predict():
    """Live IK prediction."""
    try:
        x = float(request.args.get('x', 0.4))
        y = float(request.args.get('y', 0.0))
        z = float(request.args.get('z', 0.6))
        roll = float(request.args.get('roll', 0.0))
        pitch = float(request.args.get('pitch', 3.14159))
        yaw = float(request.args.get('yaw', 0.0))

        solver = get_solver()
        if solver is None:
            return jsonify({"error": "Model not loaded"}), 500

        result = solver.solve([x, y, z, roll, pitch, yaw])
        # Add arm positions for 3D visualization
        result['arm_positions'] = solver.get_arm_positions(result['joint_angles'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/trajectory')
def api_trajectory():
    """Generate and return trajectory IK results."""
    try:
        from src.trajectory import get_trajectory

        traj_type = request.args.get('type', 'circle')
        n_points = int(request.args.get('points', 60))

        solver = get_solver()
        if solver is None:
            return jsonify({"error": "Model not loaded"}), 500

        waypoints = get_trajectory(traj_type, n_points=n_points)
        results = solver.solve_trajectory(waypoints)

        # Collect arm positions for each frame
        frames = []
        for r in results:
            frames.append({
                'arm_positions': solver.get_arm_positions(r['joint_angles']),
                'joint_angles_deg': r['joint_angles_deg'],
                'position_error_mm': r['position_error_mm'],
                'target': r['achieved_pose'][:3],
            })

        return jsonify({
            'type': traj_type,
            'n_points': n_points,
            'frames': frames,
            'avg_error_mm': float(np.mean([r['position_error_mm'] for r in results])),
            'max_error_mm': float(np.max([r['position_error_mm'] for r in results])),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/iterations')
def api_iterations():
    """Return iteration comparison data from context log."""
    ctx_path = project_root / "context_log.json"
    if ctx_path.exists():
        with open(ctx_path) as f:
            ctx = json.load(f)
        return jsonify({
            "iterations": ctx.get("iterations", []),
            "best_model": ctx.get("best_model", {}),
            "decisions": ctx.get("decisions", []),
        })
    return jsonify({"error": "No context log"}), 404


@app.route('/api/random-demo')
def api_random_demo():
    """Random pose prediction with full analysis."""
    solver = get_solver()
    if solver is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Generate a random reachable pose
    from src.robot_model import RobotModel
    robot = RobotModel()
    q_rand = robot.random_joint_config()
    target_pose = robot.forward_kinematics(q_rand)

    result = solver.solve(target_pose.tolist())
    result['arm_positions'] = solver.get_arm_positions(result['joint_angles'])
    result['target_pose'] = target_pose.tolist()
    result['ground_truth_joints_deg'] = np.degrees(q_rand).tolist()
    return jsonify(result)


@app.route('/api/plots')
def api_plots():
    """Return list of available plot images."""
    plots_dir = project_root / "web" / "static" / "plots"
    if plots_dir.exists():
        plots = [f.name for f in plots_dir.iterdir() if f.suffix in ('.png', '.gif')]
        return jsonify({"plots": sorted(plots)})
    return jsonify({"plots": []})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
