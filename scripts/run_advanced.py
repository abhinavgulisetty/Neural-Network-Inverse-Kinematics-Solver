#!/usr/bin/env python3
"""
SOTA IK Solver - Advanced Training Pipeline

This script runs the complete advanced training pipeline including:
1. Data verification (or generation if needed)
2. Training of advanced architectures (V6-V10)
3. Comprehensive evaluation
4. Ensemble creation and testing

Usage:
    python scripts/run_advanced.py --all              # Full pipeline
    python scripts/run_advanced.py --train 6 7       # Train specific models
    python scripts/run_advanced.py --evaluate         # Evaluate all models
    python scripts/run_advanced.py --ensemble         # Create and test ensemble
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_data():
    """Check if training data exists, generate if not."""
    data_dir = project_root / "data"
    required_files = ["train.npz", "val.npz", "test.npz", "normalization_params.npz"]
    
    missing = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing:
        print(f"Missing data files: {missing}")
        print("Generating training data...")
        from src.data_generator import generate_all_data
        generate_all_data()
    else:
        print("Training data found.")
        
        # Print data statistics
        import numpy as np
        train = np.load(data_dir / "train.npz")
        val = np.load(data_dir / "val.npz")
        test = np.load(data_dir / "test.npz")
        
        print(f"  Train samples: {len(train['poses']):,}")
        print(f"  Val samples: {len(val['poses']):,}")
        print(f"  Test samples: {len(test['poses']):,}")


def train_models(iterations: list, epochs: int = 200, batch_size: int = 256):
    """Train specified model iterations."""
    from src.train_advanced import train_advanced_model
    
    for iteration in iterations:
        print(f"\n{'='*60}")
        print(f"Training Model V{iteration}")
        print(f"{'='*60}")
        
        model, history = train_advanced_model(
            iteration=iteration,
            max_epochs=epochs,
            batch_size=batch_size if iteration != 9 else 128,
            lr=1e-3 if iteration != 9 else 1e-4,
            patience=30,
            use_curriculum=True,
            use_physics_loss=(iteration in [6, 7]),
            fk_loss_weight=0.5
        )
        
        print(f"Model V{iteration} training complete.")


def evaluate_models(iterations: list = None, use_tto: bool = True):
    """Evaluate trained models."""
    import torch
    from src.evaluate_advanced import evaluate_advanced_model, run_comprehensive_benchmark
    
    if iterations is None:
        # Find all available models
        model_dir = project_root / "models"
        iterations = []
        for i in range(6, 11):
            if (model_dir / f"best_model_iter{i}.pth").exists():
                iterations.append(i)
    
    if not iterations:
        print("No trained models found!")
        return
    
    print(f"\nEvaluating models: {iterations}")
    run_comprehensive_benchmark(iterations=iterations, use_tto=use_tto)


def create_and_test_ensemble():
    """Create ensemble from trained models and evaluate."""
    import torch
    from src.ensemble import create_ensemble_from_checkpoints, create_cascaded_solver
    from src.utils import Normalizer
    from src.robot_model import RobotModel
    import numpy as np
    
    model_dir = project_root / "models"
    data_dir = project_root / "data"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find available models
    available = []
    for i in range(6, 11):
        if (model_dir / f"best_model_iter{i}.pth").exists():
            available.append(i)
    
    if len(available) < 2:
        print("Need at least 2 trained models for ensemble")
        return
    
    print(f"\nCreating ensemble with models: {available}")
    
    # Create ensemble
    ensemble = create_ensemble_from_checkpoints(
        model_dir, iterations=available, device=device
    )
    
    # Load test data
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    test_data = np.load(data_dir / "test.npz")
    test_poses = test_data['poses'][:500].astype(np.float32)
    test_poses_norm = normalizer.normalize_input(test_poses)
    
    robot = RobotModel()
    
    # Test different strategies
    strategies = ['weighted_avg', 'best_of_n', 'uncertainty_weighted']
    
    print("\n=== Ensemble Evaluation ===")
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        pos_errors = []
        ensemble.eval()
        
        with torch.no_grad():
            for i in range(len(test_poses)):
                x = torch.from_numpy(test_poses_norm[i:i+1]).to(device)
                
                try:
                    pred = ensemble.predict_angles(x, strategy=strategy)
                    achieved = robot.forward_kinematics(pred[0].cpu().numpy())
                    pos_err = np.linalg.norm(achieved[:3] - test_poses[i, :3]) * 1000
                    pos_errors.append(pos_err)
                except Exception as e:
                    pos_errors.append(float('inf'))
        
        pos_errors = np.array(pos_errors)
        valid = np.isfinite(pos_errors)
        
        print(f"  Position RMSE: {np.sqrt(np.mean(pos_errors[valid]**2)):.2f} mm")
        print(f"  Position Median: {np.median(pos_errors[valid]):.2f} mm")
        print(f"  Success (5mm): {np.mean(pos_errors[valid] < 5.0) * 100:.1f}%")


def print_summary():
    """Print summary of available models and results."""
    import json
    
    print("\n" + "="*60)
    print("SOTA IK SOLVER - PROJECT SUMMARY")
    print("="*60)
    
    # List models
    from src.model import list_all_models
    list_all_models()
    
    # Check for results
    results_dir = project_root / "results"
    benchmark_path = results_dir / "comprehensive_benchmark.json"
    
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            results = json.load(f)
        
        print("\n=== Benchmark Results ===")
        
        if 'iterations' in results:
            print("\nModel Performance:")
            print(f"{'Iter':>5} {'Pos RMSE (mm)':>14} {'Success 5mm':>12} {'Inference':>12}")
            print("-" * 50)
            
            for m in results['iterations']:
                print(f"{m['iteration']:>5} {m['position_rmse_mm']:>14.2f} "
                      f"{m['success_rate_5mm_pct']:>11.1f}% "
                      f"{m['avg_inference_ms']:>10.3f}ms")
        
        if 'comparison' in results and results['comparison']:
            comp = results['comparison']
            print("\n=== Best Models ===")
            if 'best_position_accuracy' in comp:
                print(f"Best Accuracy: V{comp['best_position_accuracy']['iteration']} "
                      f"({comp['best_position_accuracy']['position_rmse_mm']:.2f} mm)")
            if 'best_success_rate' in comp:
                print(f"Best Success: V{comp['best_success_rate']['iteration']} "
                      f"({comp['best_success_rate']['success_rate_5mm_pct']:.1f}%)")
            if 'fastest_inference' in comp:
                print(f"Fastest: V{comp['fastest_inference']['iteration']} "
                      f"({comp['fastest_inference']['avg_inference_ms']:.3f} ms)")
    else:
        print("\nNo benchmark results found. Run evaluation first.")


def main():
    parser = argparse.ArgumentParser(
        description="SOTA IK Solver - Advanced Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_advanced.py --all               # Full pipeline (V6-V10)
  python scripts/run_advanced.py --train 6 7 10     # Train specific models
  python scripts/run_advanced.py --evaluate         # Evaluate all trained models
  python scripts/run_advanced.py --ensemble         # Test ensemble methods
  python scripts/run_advanced.py --summary          # Show project summary
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                        help='Run complete pipeline (data, train, evaluate)')
    parser.add_argument('--train', type=int, nargs='*', 
                        help='Train specified iterations (6-10)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate all trained models')
    parser.add_argument('--ensemble', action='store_true',
                        help='Create and test ensemble')
    parser.add_argument('--summary', action='store_true',
                        help='Print project summary')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--no_tto', action='store_true',
                        help='Disable test-time optimization in evaluation')
    
    args = parser.parse_args()
    
    # Default to summary if no args
    if not any([args.all, args.train is not None, args.evaluate, args.ensemble, args.summary]):
        args.summary = True
    
    if args.summary:
        print_summary()
        return
    
    if args.all:
        print("\n=== Running Complete Advanced Pipeline ===\n")
        check_data()
        train_models([6, 7, 8, 9, 10], args.epochs, args.batch_size)
        evaluate_models(use_tto=not args.no_tto)
        create_and_test_ensemble()
        print_summary()
        return
    
    if args.train is not None:
        iterations = args.train if args.train else [6, 7, 8, 9, 10]
        check_data()
        train_models(iterations, args.epochs, args.batch_size)
    
    if args.evaluate:
        evaluate_models(use_tto=not args.no_tto)
    
    if args.ensemble:
        create_and_test_ensemble()


if __name__ == "__main__":
    main()
