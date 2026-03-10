"""
Master script: runs the full pipeline end-to-end.
Reads context_log.json to resume from last completed phase.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_context_log, update_phase, ensure_dir


def main():
    ctx = load_context_log()
    last = ctx.get("last_completed_phase")

    print("=" * 60)
    print("NEURAL NETWORK IK SOLVER — FULL PIPELINE")
    print("=" * 60)
    print(f"Resuming from: {last or 'beginning'}\n")

    # Phase 1: Verify robot model
    if not last or last < "phase1":
        print("\n" + "=" * 60)
        print("PHASE 1: Robot Model Verification")
        print("=" * 60)
        from src.robot_model import verify_robot_model
        verify_robot_model()
        update_phase("phase1", "Robot model verified")

    # Phase 2: Data generation
    if not last or last < "phase2":
        print("\n" + "=" * 60)
        print("PHASE 2: Data Generation")
        print("=" * 60)
        from src.data_generator import generate_all_data
        generate_all_data()

    # Phase 3: Iterative training
    if not last or last < "phase3":
        print("\n" + "=" * 60)
        print("PHASE 3: Iterative Training")
        print("=" * 60)
        from src.train import run_training_iterations
        run_training_iterations(max_iterations=3)

    # Phase 4: Full evaluation + numerical benchmark
    if not last or last < "phase4":
        print("\n" + "=" * 60)
        print("PHASE 4: Full Evaluation")
        print("=" * 60)
        from src.evaluate import run_numerical_ik_benchmark, compile_all_metrics
        data_dir = project_root / "data"
        results_dir = project_root / "results"
        run_numerical_ik_benchmark(data_dir, results_dir, n_samples=500)
        compile_all_metrics(results_dir)
        update_phase("phase4", "Full evaluation complete")

    # Phase 5: Visualizations
    if not last or last < "phase5":
        print("\n" + "=" * 60)
        print("PHASE 5: Generating Visualizations")
        print("=" * 60)
        from src.visualization import generate_all_visualizations
        generate_all_visualizations()

        # Generate arm animations for demo trajectories
        try:
            from src.trajectory import get_trajectory
            from src.ik_solver import IKSolver
            from src.visualization import generate_arm_animation
            import numpy as np

            solver = IKSolver()
            plots_dir = project_root / "web" / "static" / "plots"

            for traj_type in ['circle', 'helix', 'line']:
                print(f"  Generating {traj_type} trajectory animation...")
                waypoints = get_trajectory(traj_type, n_points=60)
                results = solver.solve_trajectory(waypoints)
                joint_traj = np.array([r['joint_angles'] for r in results])
                save_path = plots_dir / f'arm_animation_{traj_type}.gif'
                generate_arm_animation(joint_traj, save_path, title=f'{traj_type.capitalize()} Trajectory')
        except Exception as e:
            print(f"  ⚠️ Animation generation failed: {e}")

        update_phase("phase5", "Visualizations generated")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo launch the web dashboard:")
    print(f"  cd '{project_root}'")
    print(f"  python scripts/run_web.py")
    print(f"  Then open http://localhost:5000")


if __name__ == "__main__":
    main()
