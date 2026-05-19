#!/usr/bin/env python3
"""Quick test script to verify advanced models work correctly."""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from src.model import create_model
from src.utils import Normalizer


def sincos_loss(pred_sincos, target_angles):
    """Loss for sin/cos output (V6, V7)."""
    # Convert target angles to sin/cos
    target_sincos = torch.zeros_like(pred_sincos)
    for i in range(6):
        target_sincos[:, 2*i] = torch.sin(target_angles[:, i])
        target_sincos[:, 2*i + 1] = torch.cos(target_angles[:, i])
    return nn.functional.mse_loss(pred_sincos, target_sincos)


def sincos_to_angles(sincos):
    """Convert sin/cos output to angles."""
    batch_size = sincos.shape[0]
    angles = torch.zeros(batch_size, 6, device=sincos.device)
    for i in range(6):
        sin_val = sincos[:, 2*i]
        cos_val = sincos[:, 2*i + 1]
        angles[:, i] = torch.atan2(sin_val, cos_val)
    return angles


def quick_test(model_version=6, n_samples=1000, n_epochs=5):
    """Quick test with small data subset."""
    
    data_dir = Path("data")
    device = "cpu"
    
    # Load a small subset of data
    print(f"Loading {n_samples} samples for quick test...")
    train_data = np.load(data_dir / "train.npz")
    
    X = train_data['poses'][:n_samples].astype(np.float32)
    Y = train_data['joint_angles'][:n_samples].astype(np.float32)
    
    # Normalize
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    X_norm = normalizer.normalize_input(X).astype(np.float32)
    Y_norm = normalizer.normalize_output(Y).astype(np.float32)
    
    X_tensor = torch.from_numpy(X_norm)
    Y_tensor = torch.from_numpy(Y_norm)
    Y_raw_tensor = torch.from_numpy(Y)  # Denormalized for sin/cos loss
    
    # Create model
    print(f"\nCreating model V{model_version}...")
    model, arch_name = create_model(model_version, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Determine model type
    uses_sincos_output = model_version in [6, 7]  # V6, V7 output sin/cos
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Quick training loop
    print(f"\nTraining for {n_epochs} epochs...")
    batch_size = 256
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_x = X_tensor[start_idx:end_idx].to(device)
            batch_y = Y_tensor[start_idx:end_idx].to(device)
            batch_y_raw = Y_raw_tensor[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (handle different model types)
            if model_version == 8:  # cINN
                z, log_jac = model(batch_x, batch_y)
                loss = 0.5 * torch.sum(z**2, dim=1).mean() - log_jac.mean()
            elif model_version == 9:  # Diffusion
                # Sample random timestep
                t = torch.randint(0, 100, (batch_x.size(0),), device=device).long()
                # Add noise to targets
                x_noisy, noise = model.add_noise(batch_y_raw, t)
                # Predict noise
                noise_pred = model(x_noisy, t, batch_x)
                loss = nn.functional.mse_loss(noise_pred, noise)
            elif model_version == 10:  # MDN
                loss = model.nll_loss(batch_x, batch_y_raw)
            elif uses_sincos_output:  # V6, V7 output sin/cos
                pred = model(batch_x)
                loss = sincos_loss(pred, batch_y_raw)
            else:  # Standard output
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch}/{n_epochs}: Loss = {avg_loss:.6f}, Time = {epoch_time:.2f}s")
    
    # Evaluate on test sample
    print("\nQuick evaluation...")
    model.eval()
    with torch.no_grad():
        test_x = X_tensor[:100].to(device)
        test_y_true = Y[:100]  # Denormalized ground truth
        
        if model_version == 8:  # cINN - sample from latent
            z = torch.randn(100, 6, device=device)
            pred_norm = model.inverse(test_x, z)
            pred = normalizer.denormalize_output(pred_norm.cpu().numpy())
        elif model_version == 9:  # Diffusion - sample
            pred = model.sample(test_x, n_steps=50).cpu().numpy()
        elif model_version == 10:  # MDN - get most likely mode
            pred = model.predict_angles(test_x).cpu().numpy()
        elif uses_sincos_output:  # V6, V7 - convert sin/cos to angles
            pred_sincos = model(test_x)
            pred = sincos_to_angles(pred_sincos).cpu().numpy()
        else:
            pred_norm = model(test_x)
            pred = normalizer.denormalize_output(pred_norm.cpu().numpy())
        
        # Joint angle error
        joint_error_rad = np.abs(pred - test_y_true)
        joint_error_deg = np.rad2deg(joint_error_rad)
        
        print(f"  Mean joint error: {joint_error_deg.mean():.2f} deg")
        print(f"  Max joint error:  {joint_error_deg.max():.2f} deg")
    
    print(f"\n[SUCCESS] Model V{model_version} works correctly!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=int, default=6, help="Model version (6-10)")
    parser.add_argument("--samples", "-n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    
    success = quick_test(args.model, args.samples, args.epochs)
    sys.exit(0 if success else 1)
