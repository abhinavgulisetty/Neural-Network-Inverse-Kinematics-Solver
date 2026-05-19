#!/usr/bin/env python3

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
from typing import Tuple, Optional

PUMA_DH = {
    'a2': 0.4318,
    'a3': 0.0203,
    'd3': 0.1244,
    'd4': 0.4318,
}


class DifferentiableFK(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.register_buffer('dh_a', torch.tensor([0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0]))
        self.register_buffer('dh_d', torch.tensor([0.6718, 0.0, 0.1500, 0.4318, 0.0, 0.0]))
        self.register_buffer('dh_alpha', torch.tensor([np.pi/2, 0.0, -np.pi/2, np.pi/2, -np.pi/2, 0.0]))
        
    def dh_matrix(self, theta, d, a, alpha):
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        
        batch_size = theta.shape[0]
        T = torch.zeros(batch_size, 4, 4, device=theta.device, dtype=theta.dtype)
        
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st * ca
        T[:, 0, 2] = st * sa
        T[:, 0, 3] = a * ct
        
        T[:, 1, 0] = st
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -ct * sa
        T[:, 1, 3] = a * st
        
        T[:, 2, 1] = sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = d
        
        T[:, 3, 3] = 1.0
        
        return T
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]
        device = q.device
        dtype = q.dtype
        
        T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        for i in range(6):
            a = self.dh_a[i].expand(batch_size)
            d = self.dh_d[i].expand(batch_size)
            alpha = self.dh_alpha[i].expand(batch_size)
            
            Ti = self.dh_matrix(q[:, i], d, a, alpha)
            T = T @ Ti
        
        pos = T[:, :3, 3]
        
        R = T[:, :3, :3]
        
        pitch = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2 + 1e-8))
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        
        pose = torch.cat([pos, roll.unsqueeze(1), pitch.unsqueeze(1), yaw.unsqueeze(1)], dim=1)
        
        return pose


class OptimizedIKNet(nn.Module):
    
    def __init__(self, hidden_dim: int = 512, n_layers: int = 6, 
                 n_frequencies: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.n_frequencies = n_frequencies
        
        B = torch.randn(6, n_frequencies) * 10.0
        self.register_buffer('B', B)
        
        input_dim = 6 + 2 * n_frequencies
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ))
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 12),
        )
        
        self.fk = DifferentiableFK()
        
        self.register_buffer('joint_min', torch.tensor([-2.79, -3.93, -0.79, -4.71, -1.92, -4.71]))
        self.register_buffer('joint_max', torch.tensor([2.79, 0.79, 3.93, 4.71, 1.92, 4.71]))
    
    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        B = self.B.to(x.dtype)
        x_proj = 2 * np.pi * x @ B
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        x = self.encode_input(pose)
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = h + block(h)
        
        return self.output_head(h)
    
    def predict_angles(self, pose: torch.Tensor) -> torch.Tensor:
        sincos = self.forward(pose)
        angles = torch.zeros(pose.shape[0], 6, device=pose.device, dtype=pose.dtype)
        for i in range(6):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles
    
    def clamp_joints(self, q: torch.Tensor) -> torch.Tensor:
        return torch.clamp(q, self.joint_min, self.joint_max)
    
    @torch.enable_grad()
    def refine_tto(self, pose: torch.Tensor, q_init: torch.Tensor, 
                   n_steps: int = 10, lr: float = 0.1,
                   pos_weight: float = 1.0, ori_weight: float = 0.1) -> torch.Tensor:
        q = q_init.clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([q], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        best_q = q_init.clone()
        best_error = torch.full((pose.shape[0],), float('inf'), device=pose.device)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            pred_pose = self.fk(q)
            
            pos_error = torch.sum((pred_pose[:, :3] - pose[:, :3])**2, dim=1)
            
            ori_diff = pred_pose[:, 3:] - pose[:, 3:]
            ori_diff = torch.atan2(torch.sin(ori_diff), torch.cos(ori_diff))
            ori_error = torch.sum(ori_diff**2, dim=1)
            
            sample_loss = pos_weight * pos_error + ori_weight * ori_error
            loss = sample_loss.mean()
            
            with torch.no_grad():
                improved = sample_loss < best_error
                best_error[improved] = sample_loss[improved]
                best_q[improved] = q[improved].clone()
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            with torch.no_grad():
                q.data = self.clamp_joints(q.data)
        
        return best_q.detach()
    
    @torch.enable_grad()
    def refine_tto_fast(self, pose: torch.Tensor, q_init: torch.Tensor,
                        n_steps: int = 30, lr: float = 1.5,
                        early_stop_mm: float = 0.5) -> torch.Tensor:
        q = q_init.clone().detach().requires_grad_(True)
        
        best_q = q_init.clone()
        best_error = torch.full((pose.shape[0],), float('inf'), device=pose.device)
        
        early_stop_m = early_stop_mm / 1000.0
        
        for step in range(n_steps):
            pred_pose = self.fk(q)
            
            pos_error = torch.sum((pred_pose[:, :3] - pose[:, :3])**2, dim=1)
            pos_error_m = torch.sqrt(pos_error)
            loss = pos_error.mean()
            
            with torch.no_grad():
                improved = pos_error_m < best_error
                best_error[improved] = pos_error_m[improved]
                best_q[improved] = q[improved].clone()
                
                if (pos_error_m < early_stop_m).all():
                    break
            
            loss.backward()
            
            with torch.no_grad():
                q.data = q.data - lr * q.grad
                q.data = self.clamp_joints(q.data)
                q.grad.zero_()
        
        return best_q.detach()
    
    @torch.enable_grad()
    def refine_tto_adaptive(self, pose: torch.Tensor, q_init: torch.Tensor,
                            target_error_mm: float = 1.0, max_steps: int = 1000,
                            lr: float = 0.05) -> torch.Tensor:
        q = q_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([q], lr=lr)
        
        best_q = q_init.clone()
        best_error = torch.full((pose.shape[0],), float('inf'), device=pose.device)
        
        target_error_m = target_error_mm / 1000.0
        
        for step in range(max_steps):
            optimizer.zero_grad()
            
            pred_pose = self.fk(q)
            pos_error = torch.sqrt(torch.sum((pred_pose[:, :3] - pose[:, :3])**2, dim=1))
            
            with torch.no_grad():
                improved = pos_error < best_error
                best_error[improved] = pos_error[improved]
                best_q[improved] = q[improved].clone()
            
            if (best_error < target_error_m).all():
                break
            
            ori_diff = pred_pose[:, 3:] - pose[:, 3:]
            ori_diff = torch.atan2(torch.sin(ori_diff), torch.cos(ori_diff))
            loss = (pos_error**2 + 0.1 * torch.sum(ori_diff**2, dim=1)).mean()
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                q.data = self.clamp_joints(q.data)
        
        return best_q.detach()
    
    def solve(self, pose: torch.Tensor, use_tto: bool = True, 
              tto_steps: int = 20, tto_lr: float = 0.05) -> torch.Tensor:
        q_init = self.predict_angles(pose)
        
        if use_tto:
            q = self.refine_tto(pose, q_init, n_steps=tto_steps, lr=tto_lr)
        else:
            q = q_init
        
        return q
    
    def solve_fast(self, pose: torch.Tensor, tto_steps: int = 30, 
                   tto_lr: float = 1.5, early_stop_mm: float = 0.5) -> torch.Tensor:
        q_init = self.predict_angles(pose)
        
        q = self.refine_tto_fast(pose, q_init, n_steps=tto_steps, lr=tto_lr, 
                                  early_stop_mm=early_stop_mm)
        return q
    
    def solve_precise(self, pose: torch.Tensor, target_error_mm: float = 1.0,
                      max_restarts: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = pose.shape[0]
        device = pose.device
        
        q_init = self.predict_angles(pose)
        
        q_best = self.refine_tto(pose, q_init, n_steps=500, lr=0.05)
        
        with torch.no_grad():
            pred_pose = self.fk(q_best)
            error_mm = torch.norm(pred_pose[:, :3] - pose[:, :3], dim=1) * 1000
        
        needs_work = error_mm > target_error_mm
        
        if needs_work.any():
            for restart in range(max_restarts):
                if not needs_work.any():
                    break
                    
                q_perturbed = q_best.clone()
                noise = torch.randn_like(q_best[needs_work]) * 0.3
                q_perturbed[needs_work] = self.clamp_joints(q_best[needs_work] + noise)
                
                q_refined = self.refine_tto(pose[needs_work], q_perturbed[needs_work], 
                                           n_steps=300, lr=0.03)
                
                with torch.no_grad():
                    refined_pose = self.fk(q_refined)
                    refined_error = torch.norm(refined_pose[:, :3] - pose[needs_work, :3], dim=1) * 1000
                    
                    improved = refined_error < error_mm[needs_work]
                    indices_need_work = torch.where(needs_work)[0]
                    
                    for i, idx in enumerate(indices_need_work):
                        if improved[i]:
                            q_best[idx] = q_refined[i]
                            error_mm[idx] = refined_error[i]
                    
                    needs_work = error_mm > target_error_mm
        
        return q_best, error_mm


def sincos_loss(pred_sincos: torch.Tensor, target_angles: torch.Tensor) -> torch.Tensor:
    target_sincos = torch.zeros_like(pred_sincos)
    for i in range(6):
        target_sincos[:, 2*i] = torch.sin(target_angles[:, i])
        target_sincos[:, 2*i + 1] = torch.cos(target_angles[:, i])
    return nn.functional.mse_loss(pred_sincos, target_sincos)


def train_optimized_model(n_epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    data_dir = Path("data")
    
    print("Loading data...")
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")
    
    X_train = torch.from_numpy(train_data['poses'].astype(np.float32)).to(device)
    Y_train = torch.from_numpy(train_data['joint_angles'].astype(np.float32)).to(device)
    X_val = torch.from_numpy(val_data['poses'].astype(np.float32)).to(device)
    Y_val = torch.from_numpy(val_data['joint_angles'].astype(np.float32)).to(device)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
    model = OptimizedIKNet(hidden_dim=512, n_layers=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size
    
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        perm = torch.randperm(n_train, device=device)
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]
        
        model.train()
        train_loss = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_train)
            
            batch_x = X_train_shuffled[start_idx:end_idx]
            batch_y = Y_train_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            pred_sincos = model(batch_x)
            loss = sincos_loss(pred_sincos, batch_y)
            
            if epoch % 5 == 0:
                pred_angles = model.predict_angles(batch_x)
                pred_pose = model.fk(pred_angles)
                fk_loss = nn.functional.mse_loss(pred_pose[:, :3], batch_x[:, :3])
                loss = loss + 0.1 * fk_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= n_batches
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = sincos_loss(val_pred, Y_val).item()
            
            val_angles = model.predict_angles(X_val[:1000])
            val_poses = model.fk(val_angles)
            pos_error = torch.norm(val_poses[:, :3] - X_val[:1000, :3], dim=1) * 1000
            mean_pos_error = pos_error.mean().item()
        
        epoch_time = time.time() - epoch_start
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            marker = " *"
        else:
            marker = ""
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Pos Err: {mean_pos_error:.2f}mm | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {epoch_time:.1f}s{marker}")
    
    model.load_state_dict(best_model_state)
    
    save_path = Path("models") / "optimized_ik_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_dim': 512,
            'n_layers': 6,
            'n_frequencies': 32,
        }
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    return model


def evaluate_optimized_model(model: OptimizedIKNet, use_tto: bool = True, 
                              tto_steps: int = 20):
    
    device = next(model.parameters()).device
    
    test_data = np.load("data/test.npz")
    X_test = torch.from_numpy(test_data['poses'].astype(np.float32)).to(device)
    Y_test = test_data['joint_angles'].astype(np.float32)
    
    n_test = min(1000, len(X_test))
    X_test = X_test[:n_test]
    Y_test = Y_test[:n_test]
    
    model.eval()
    
    print("\n" + "="*60)
    print("EVALUATION WITHOUT TTO (NN only)")
    print("="*60)
    
    start = time.time()
    with torch.no_grad():
        pred_angles_no_tto = model.predict_angles(X_test)
        pred_poses_no_tto = model.fk(pred_angles_no_tto)
    time_no_tto = (time.time() - start) / n_test * 1000
    
    pos_error_no_tto = torch.norm(pred_poses_no_tto[:, :3] - X_test[:, :3], dim=1) * 1000
    
    print(f"  Position error: {pos_error_no_tto.mean():.4f}mm (mean), {pos_error_no_tto.max():.4f}mm (max)")
    print(f"  Sub-1mm rate: {(pos_error_no_tto < 1.0).sum().item()}/{n_test} ({(pos_error_no_tto < 1.0).float().mean()*100:.1f}%)")
    print(f"  Time per sample: {time_no_tto:.3f}ms")
    
    print("\n" + "="*60)
    print(f"EVALUATION WITH TTO ({tto_steps} steps)")
    print("="*60)
    
    start = time.time()
    pred_angles_tto = model.solve(X_test, use_tto=True, tto_steps=tto_steps)
    with torch.no_grad():
        pred_poses_tto = model.fk(pred_angles_tto)
    time_tto = (time.time() - start) / n_test * 1000
    
    pos_error_tto = torch.norm(pred_poses_tto[:, :3] - X_test[:, :3], dim=1) * 1000
    
    print(f"  Position error: {pos_error_tto.mean():.4f}mm (mean), {pos_error_tto.max():.4f}mm (max)")
    print(f"  Sub-1mm rate: {(pos_error_tto < 1.0).sum().item()}/{n_test} ({(pos_error_tto < 1.0).float().mean()*100:.1f}%)")
    print(f"  Time per sample: {time_tto:.3f}ms")
    
    print("\n" + "="*60)
    print("ERROR DISTRIBUTION (with TTO)")
    print("="*60)
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for t in thresholds:
        pct = (pos_error_tto < t).float().mean() * 100
        print(f"  <{t:5.1f}mm: {pct:6.2f}%")
    
    return {
        'no_tto': {
            'mean_error_mm': pos_error_no_tto.mean().item(),
            'max_error_mm': pos_error_no_tto.max().item(),
            'time_ms': time_no_tto,
        },
        'with_tto': {
            'mean_error_mm': pos_error_tto.mean().item(),
            'max_error_mm': pos_error_tto.max().item(),
            'time_ms': time_tto,
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--tto-steps", type=int, default=20, help="TTO refinement steps")
    args = parser.parse_args()
    
    if args.train:
        model = train_optimized_model(n_epochs=args.epochs)
        evaluate_optimized_model(model, tto_steps=args.tto_steps)
    elif args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load("models/optimized_ik_model.pth", map_location=device)
        
        config = checkpoint.get('config', {'hidden_dim': 512, 'n_layers': 6, 'n_frequencies': 32})
        model = OptimizedIKNet(**config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        evaluate_optimized_model(model, tto_steps=args.tto_steps)
    else:
        print("Usage: python optimized_ik.py --train [--epochs N]")
        print("       python optimized_ik.py --eval [--tto-steps N]")
