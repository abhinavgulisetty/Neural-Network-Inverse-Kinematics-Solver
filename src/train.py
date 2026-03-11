"""
Training loop with early stopping, LR scheduling, and checkpoint saving.
Optimized for CPU training.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import create_dataloaders
from src.model import create_model, IKNetV4
from src.utils import (
    load_context_log, save_context_log, log_iteration,
    update_phase, ensure_dir, Normalizer
)
from src.evaluate import evaluate_model


def train_model(iteration, data_dir=None, model_dir=None, max_epochs=100,
                batch_size=256, lr=1e-3, patience=25, device='cpu'):
    """
    Train a model for one iteration of the experiment.

    Args:
        iteration: which architecture to use (1-5)
        data_dir: path to data directory
        model_dir: path to save models
        max_epochs: maximum training epochs
        batch_size: training batch size
        lr: initial learning rate
        patience: early stopping patience
        device: 'cpu' (we are CPU-only)

    Returns:
        model, training_history dict
    """
    project_root = Path(__file__).parent.parent
    if data_dir is None:
        data_dir = project_root / "data"
    if model_dir is None:
        model_dir = project_root / "models"
    ensure_dir(model_dir)

    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    print(f"\n{'='*60}")
    print(f"TRAINING ITERATION {iteration}")
    print(f"{'='*60}")
    print(f"Device: CPU ({n_threads} threads)")

    print("\nLoading data...")
    loaders = create_dataloaders(str(data_dir), batch_size=batch_size)

    model, arch_name = create_model(iteration)
    model = model.to(device)

    is_sincos = isinstance(model, IKNetV4)
    if is_sincos:
        def sincos_loss(pred, target_angles):
            """MSE on sin/cos representation of target angles."""
            target_sincos = torch.zeros(target_angles.shape[0], 12, device=device)
            for i in range(6):
                target_sincos[:, 2*i] = torch.sin(target_angles[:, i])
                target_sincos[:, 2*i + 1] = torch.cos(target_angles[:, i])
            return nn.MSELoss()(pred, target_sincos)
        criterion = sincos_loss
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch_time': []
    }

    checkpoint_path = model_dir / f"best_model_iter{iteration}.pth"

    print(f"\nStarting training (max {max_epochs} epochs, patience {patience})...\n")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loaders['train']:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)

            if is_sincos:
                loss = criterion(pred, batch_y)
            else:
                loss = criterion(pred, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in loaders['val']:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                if is_sincos:
                    loss = criterion(pred, batch_y)
                else:
                    loss = criterion(pred, batch_y)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'iteration': iteration,
                'architecture': arch_name,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"No improve: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n  Best val_loss: {best_val_loss:.6f} at epoch {checkpoint['epoch']}")
    print(f"  Model saved to: {checkpoint_path}")

    return model, history


def run_training_iterations(max_iterations=5):
    """
    Run the iterative training loop.
    Checks context_log to resume from where it left off.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    results_dir = project_root / "results"
    ensure_dir(results_dir)

    ctx = load_context_log()
    completed_iterations = [it['iteration'] for it in ctx.get('iterations', [])]

    for iteration in range(1, max_iterations + 1):
        if iteration in completed_iterations:
            print(f"\n  Iteration {iteration} already completed, skipping...")
            continue

        model, history = train_model(iteration, data_dir, model_dir)

        history_path = results_dir / f"training_history_iter{iteration}.json"
        serializable_history = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        print(f"\n  Evaluating iteration {iteration}...")
        metrics = evaluate_model(
            model, iteration, data_dir, results_dir,
            is_sincos=isinstance(model, IKNetV4)
        )

        hyperparams = {
            "batch_size": 256,
            "learning_rate": 1e-3,
            "max_epochs": 300,
            "patience": 25,
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "grad_clip": 1.0
        }

        from src.model import get_architecture_description
        arch_desc = get_architecture_description(iteration)

        if iteration == 1:
            changes = "Baseline MLP with 4x256 hidden layers, ReLU, Dropout 0.2"
        elif iteration == 2:
            changes = "Added BatchNorm, increased to 5 layers (512 to 128), reduced dropout to 0.15"
        elif iteration == 3:
            changes = "Added residual/skip connections for better gradient flow"
        elif iteration == 4:
            changes = "Switched to sin/cos output encoding to handle angle wrapping"
        elif iteration == 5:
            changes = "Split into position head (joints 1-3) and orientation head (joints 4-6)"

        next_steps = "Try next architecture" if iteration < max_iterations else "Complete evaluation"

        log_iteration(iteration, arch_desc, hyperparams, metrics, changes, next_steps)

        pos_rmse = metrics.get('position_rmse_mm', float('inf'))
        success_rate = metrics.get('success_rate_pct', 0)
        print(f"\n  Iteration {iteration} results:")
        print(f"    Position RMSE: {pos_rmse:.4f} mm (target: < 1.0 mm)")
        print(f"    Success rate: {success_rate:.1f}% (target: > 95%)")

        if pos_rmse < 1.0 and success_rate > 95:
            print(f"\n  Targets met at iteration {iteration}! Stopping early.")
            break

    update_phase("phase3", "Iterative training complete")


if __name__ == "__main__":
    run_training_iterations()
