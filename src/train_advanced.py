import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import create_dataloaders, IKDataset
from src.models_advanced import (
    create_advanced_model, 
    ADVANCED_MODEL_REGISTRY,
    IKNetV6, IKNetV7, IKNetV8, IKNetV9, IKNetV10,
    DifferentiableFKLayer
)
from src.utils import (
    load_context_log, save_context_log, log_iteration,
    update_phase, ensure_dir, Normalizer
)
from src.robot_model import RobotModel


class PhysicsInformedLoss(nn.Module):
    
    def __init__(self, fk_weight: float = 1.0, limit_weight: float = 0.1,
                 smooth_weight: float = 0.01, use_sincos: bool = True):
        super().__init__()
        self.fk_weight = fk_weight
        self.limit_weight = limit_weight
        self.smooth_weight = smooth_weight
        self.use_sincos = use_sincos
        
        self.fk_layer = DifferentiableFKLayer()
        
        self.register_buffer('joint_lower', torch.tensor(
            np.radians([-90, -90, 0, -90, -90, -90]), dtype=torch.float32
        ))
        self.register_buffer('joint_upper', torch.tensor(
            np.radians([90, 0, 90, 90, 90, 90]), dtype=torch.float32
        ))
    
    def forward(self, pred_sincos: torch.Tensor, target_joints: torch.Tensor,
                target_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        
        pred_joints = self._sincos_to_angles(pred_sincos)
        
        target_sincos = self._angles_to_sincos(target_joints)
        joint_loss = F.mse_loss(pred_sincos, target_sincos)
        losses['joint_loss'] = joint_loss
        
        if self.fk_weight > 0:
            pred_pose = self.fk_layer(pred_joints)
            
            pos_loss = F.mse_loss(pred_pose[:, :3], target_pose[:, :3])
            
            pred_ori_sincos = torch.cat([
                torch.sin(pred_pose[:, 3:]),
                torch.cos(pred_pose[:, 3:])
            ], dim=1)
            target_ori_sincos = torch.cat([
                torch.sin(target_pose[:, 3:]),
                torch.cos(target_pose[:, 3:])
            ], dim=1)
            ori_loss = F.mse_loss(pred_ori_sincos, target_ori_sincos)
            
            fk_loss = pos_loss + ori_loss
            losses['fk_loss'] = fk_loss
            losses['pos_loss'] = pos_loss
            losses['ori_loss'] = ori_loss
        
        if self.limit_weight > 0:
            lower_violation = F.relu(self.joint_lower - pred_joints)
            upper_violation = F.relu(pred_joints - self.joint_upper)
            limit_loss = (lower_violation.pow(2) + upper_violation.pow(2)).mean()
            losses['limit_loss'] = limit_loss
        
        total = joint_loss
        if self.fk_weight > 0:
            total = total + self.fk_weight * fk_loss
        if self.limit_weight > 0:
            total = total + self.limit_weight * limit_loss
        
        losses['total'] = total
        return losses
    
    def _sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        angles = torch.zeros(sincos.shape[0], 6, device=sincos.device)
        for i in range(6):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles
    
    def _angles_to_sincos(self, angles: torch.Tensor) -> torch.Tensor:
        sincos = torch.zeros(angles.shape[0], 12, device=angles.device)
        for i in range(6):
            sincos[:, 2*i] = torch.sin(angles[:, i])
            sincos[:, 2*i + 1] = torch.cos(angles[:, i])
        return sincos


class CInnLoss(nn.Module):
    
    def forward(self, z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        
        log_likelihood = log_pz + log_det
        
        return -log_likelihood.mean()


class DiffusionLoss(nn.Module):
    
    def forward(self, pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_noise, true_noise)


class WarmupCosineScheduler:
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 total_steps: int, lr_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        self.lr_max = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr = self.lr_max * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class CurriculumDataset(IKDataset):
    
    def __init__(self, data_path: str, normalize: bool = True, 
                 norm_params_path: Optional[str] = None):
        super().__init__(data_path, normalize, norm_params_path)
        
        self.difficulty_scores = self._compute_difficulty()
        
        self.sorted_indices = np.argsort(self.difficulty_scores)
        
        self.curriculum_fraction = 0.3
    
    def _compute_difficulty(self) -> np.ndarray:
        joints = self.joint_angles.numpy()
        
        home_dist = np.linalg.norm(joints, axis=1)
        
        singularity_score = np.abs(joints[:, 1]) + np.abs(joints[:, 2]) + np.abs(joints[:, 4])
        singularity_score = 1.0 / (singularity_score + 0.1)
        
        joint_range = np.array([np.pi, np.pi/2, np.pi/2, np.pi, np.pi, np.pi])
        boundary_dist = np.minimum(
            np.abs(joints - (-joint_range)),
            np.abs(joints - joint_range)
        ).min(axis=1)
        boundary_score = 1.0 / (boundary_dist + 0.1)
        
        difficulty = 0.4 * home_dist + 0.4 * singularity_score + 0.2 * boundary_score
        
        difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min() + 1e-8)
        
        return difficulty
    
    def set_curriculum_fraction(self, fraction: float):
        self.curriculum_fraction = np.clip(fraction, 0.1, 1.0)
    
    def __len__(self):
        return int(len(self.poses) * self.curriculum_fraction)
    
    def __getitem__(self, idx):
        actual_idx = self.sorted_indices[idx]
        return self.poses[actual_idx], self.joint_angles[actual_idx]


class HardExampleMiner:
    
    def __init__(self, buffer_size: int = 10000, hard_fraction: float = 0.3):
        self.buffer_size = buffer_size
        self.hard_fraction = hard_fraction
        
        self.loss_buffer = []
        self.sample_indices = []
    
    def update(self, indices: torch.Tensor, losses: torch.Tensor):
        for idx, loss in zip(indices.tolist(), losses.tolist()):
            if len(self.loss_buffer) >= self.buffer_size:
                self.loss_buffer.pop(0)
                self.sample_indices.pop(0)
            
            self.loss_buffer.append(loss)
            self.sample_indices.append(idx)
    
    def get_hard_indices(self, n_samples: int) -> List[int]:
        if len(self.loss_buffer) == 0:
            return []
        
        sorted_idx = np.argsort(self.loss_buffer)[::-1]
        n_hard = min(n_samples, int(len(sorted_idx) * self.hard_fraction))
        
        return [self.sample_indices[i] for i in sorted_idx[:n_hard]]


class TestTimeOptimizer:
    
    def __init__(self, robot_model: RobotModel, n_steps: int = 10, 
                 lr: float = 0.01, device: str = 'cpu'):
        self.robot = robot_model
        self.n_steps = n_steps
        self.lr = lr
        self.device = device
        self.fk_layer = DifferentiableFKLayer().to(device)
    
    def refine(self, initial_joints: torch.Tensor, 
               target_pose: torch.Tensor) -> torch.Tensor:
        joints = initial_joints.clone().requires_grad_(True)
        
        optimizer = optim.Adam([joints], lr=self.lr)
        
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            
            achieved_pose = self.fk_layer(joints)
            
            pos_error = F.mse_loss(achieved_pose[:, :3], target_pose[:, :3])
            
            pred_ori = torch.cat([torch.sin(achieved_pose[:, 3:]), 
                                  torch.cos(achieved_pose[:, 3:])], dim=1)
            target_ori = torch.cat([torch.sin(target_pose[:, 3:]), 
                                    torch.cos(target_pose[:, 3:])], dim=1)
            ori_error = F.mse_loss(pred_ori, target_ori)
            
            loss = pos_error + ori_error
            loss.backward()
            
            optimizer.step()
        
        return joints.detach()


def train_advanced_model(
    iteration: int,
    data_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    max_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 30,
    use_curriculum: bool = True,
    use_physics_loss: bool = True,
    fk_loss_weight: float = 0.5,
    use_mixed_precision: bool = True,
    device: str = 'auto'
) -> Tuple[nn.Module, Dict]:
    project_root = Path(__file__).parent.parent
    if data_dir is None:
        data_dir = project_root / "data"
    if model_dir is None:
        model_dir = project_root / "models"
    ensure_dir(model_dir)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
    
    print(f"\n{'='*60}")
    print(f"ADVANCED TRAINING - ITERATION {iteration}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Physics Loss: {use_physics_loss}, FK Weight: {fk_loss_weight}")
    print(f"Curriculum Learning: {use_curriculum}")
    print(f"Mixed Precision: {use_mixed_precision and device == 'cuda'}")
    
    normalizer = Normalizer()
    normalizer.load(str(data_dir / "normalization_params.npz"))
    
    print("\nLoading data...")
    loaders = create_dataloaders(str(data_dir), batch_size=batch_size)
    
    model, arch_name = create_advanced_model(iteration, device)
    
    is_transformer = isinstance(model, IKNetV6)
    is_pinn = isinstance(model, IKNetV7)
    is_cinn = isinstance(model, IKNetV8)
    is_diffusion = isinstance(model, IKNetV9)
    is_mdn = isinstance(model, IKNetV10)
    
    if use_physics_loss and (is_transformer or is_pinn):
        criterion = PhysicsInformedLoss(
            fk_weight=fk_loss_weight,
            limit_weight=0.1,
            use_sincos=True
        ).to(device)
    elif is_cinn:
        criterion = CInnLoss()
    elif is_diffusion:
        criterion = DiffusionLoss()
    elif is_mdn:
        criterion = None
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    total_steps = max_epochs * len(loaders['train'])
    warmup_steps = int(0.05 * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler() if (use_mixed_precision and device == 'cuda') else None
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'val_loss': [], 'lr': [], 'epoch_time': [],
        'fk_loss': [], 'pos_error_mm': []
    }
    
    checkpoint_path = model_dir / f"best_model_iter{iteration}.pth"
    
    curriculum_fraction = 0.3 if use_curriculum else 1.0
    
    print(f"\nStarting training (max {max_epochs} epochs, patience {patience})...\n")
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        
        if use_curriculum:
            curriculum_fraction = min(1.0, 0.3 + 0.7 * (epoch / (max_epochs * 0.7)))
        
        model.train()
        train_loss = 0.0
        train_fk_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in loaders['train']:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            batch_y_denorm = torch.tensor(
                normalizer.denormalize_output(batch_y.cpu().numpy()),
                device=device, dtype=torch.float32
            )
            batch_x_denorm = torch.tensor(
                normalizer.denormalize_input(batch_x.cpu().numpy()),
                device=device, dtype=torch.float32
            )
            
            optimizer.zero_grad()
            
            if is_diffusion:
                t = torch.randint(0, model.n_timesteps, (batch_x.shape[0],), device=device)
                x_noisy, noise = model.add_noise(batch_y_denorm, t)
                
                if scaler is not None:
                    with autocast():
                        pred_noise = model(x_noisy, t, batch_x_denorm)
                        loss = criterion(pred_noise, noise)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_noise = model(x_noisy, t, batch_x_denorm)
                    loss = criterion(pred_noise, noise)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
            elif is_cinn:
                if scaler is not None:
                    with autocast():
                        z, log_det = model(batch_y_denorm, batch_x_denorm)
                        loss = criterion(z, log_det)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z, log_det = model(batch_y_denorm, batch_x_denorm)
                    loss = criterion(z, log_det)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
            elif is_mdn:
                if scaler is not None:
                    with autocast():
                        loss = model.nll_loss(batch_x, batch_y_denorm)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model.nll_loss(batch_x, batch_y_denorm)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
            else:
                if scaler is not None:
                    with autocast():
                        pred = model(batch_x)
                        if use_physics_loss:
                            losses = criterion(pred, batch_y_denorm, batch_x_denorm)
                            loss = losses['total']
                            fk_loss = losses.get('fk_loss', torch.tensor(0.0))
                        else:
                            target_sincos = torch.zeros_like(pred)
                            for i in range(6):
                                target_sincos[:, 2*i] = torch.sin(batch_y_denorm[:, i])
                                target_sincos[:, 2*i + 1] = torch.cos(batch_y_denorm[:, i])
                            loss = criterion(pred, target_sincos)
                            fk_loss = torch.tensor(0.0)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(batch_x)
                    if use_physics_loss:
                        losses = criterion(pred, batch_y_denorm, batch_x_denorm)
                        loss = losses['total']
                        fk_loss = losses.get('fk_loss', torch.tensor(0.0))
                    else:
                        target_sincos = torch.zeros_like(pred)
                        for i in range(6):
                            target_sincos[:, 2*i] = torch.sin(batch_y_denorm[:, i])
                            target_sincos[:, 2*i + 1] = torch.cos(batch_y_denorm[:, i])
                        loss = criterion(pred, target_sincos)
                        fk_loss = torch.tensor(0.0)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            if 'fk_loss' in dir():
                train_fk_loss += fk_loss.item() if torch.is_tensor(fk_loss) else fk_loss
            n_batches += 1
        
        train_loss /= n_batches
        train_fk_loss /= n_batches
        
        model.eval()
        val_loss = 0.0
        val_pos_error = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loaders['val']:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                batch_y_denorm = torch.tensor(
                    normalizer.denormalize_output(batch_y.cpu().numpy()),
                    device=device, dtype=torch.float32
                )
                batch_x_denorm = torch.tensor(
                    normalizer.denormalize_input(batch_x.cpu().numpy()),
                    device=device, dtype=torch.float32
                )
                
                if is_diffusion:
                    pred_joints = model.predict_angles(batch_x_denorm, n_steps=50)
                elif is_cinn:
                    pred_joints = model.predict_angles(batch_x_denorm)
                elif is_mdn:
                    pred_joints = model.predict_angles(batch_x)
                else:
                    pred_sincos = model(batch_x)
                    pred_joints = torch.zeros(batch_x.shape[0], 6, device=device)
                    for i in range(6):
                        pred_joints[:, i] = torch.atan2(pred_sincos[:, 2*i], pred_sincos[:, 2*i + 1])
                
                joint_error = F.mse_loss(pred_joints, batch_y_denorm)
                val_loss += joint_error.item()
                
                fk_layer = DifferentiableFKLayer().to(device)
                pred_pose = fk_layer(pred_joints)
                pos_error = torch.norm(pred_pose[:, :3] - batch_x_denorm[:, :3], dim=1).mean()
                val_pos_error += pos_error.item() * 1000
                
                n_val_batches += 1
        
        val_loss /= n_val_batches
        val_pos_error /= n_val_batches
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(scheduler.get_lr())
        history['epoch_time'].append(epoch_time)
        history['fk_loss'].append(train_fk_loss)
        history['pos_error_mm'].append(val_pos_error)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_pos_error_mm': val_pos_error,
                'iteration': iteration,
                'architecture': arch_name,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Pos Err: {val_pos_error:.2f}mm | "
                  f"LR: {scheduler.get_lr():.2e} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"No improve: {epochs_no_improve}/{patience}")
        
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n  Best val_loss: {best_val_loss:.6f} at epoch {checkpoint['epoch']}")
    print(f"  Best pos error: {checkpoint.get('val_pos_error_mm', 'N/A'):.2f} mm")
    print(f"  Model saved to: {checkpoint_path}")
    
    return model, history


def run_advanced_training_iterations(start_iteration: int = 6, end_iteration: int = 10):
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    results_dir = project_root / "results"
    ensure_dir(results_dir)
    
    ctx = load_context_log()
    completed_iterations = [it['iteration'] for it in ctx.get('iterations', [])]
    
    for iteration in range(start_iteration, end_iteration + 1):
        if iteration in completed_iterations:
            print(f"\n  Iteration {iteration} already completed, skipping...")
            continue
        
        model, history = train_advanced_model(
            iteration,
            data_dir,
            model_dir,
            max_epochs=200,
            batch_size=256 if iteration != 9 else 128,
            lr=1e-3 if iteration != 9 else 1e-4,
            patience=30,
            use_curriculum=True,
            use_physics_loss=(iteration in [6, 7]),
            fk_loss_weight=0.5
        )
        
        history_path = results_dir / f"training_history_iter{iteration}.json"
        serializable_history = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        from src.evaluate_advanced import evaluate_advanced_model
        
        print(f"\n  Evaluating iteration {iteration}...")
        metrics = evaluate_advanced_model(
            model, iteration, data_dir, results_dir
        )
        
        hyperparams = {
            "batch_size": 256 if iteration != 9 else 128,
            "learning_rate": 1e-3 if iteration != 9 else 1e-4,
            "max_epochs": 200,
            "patience": 30,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "WarmupCosine",
            "physics_loss": iteration in [6, 7],
            "curriculum_learning": True
        }
        
        from src.models_advanced import get_advanced_architecture_description
        arch_desc = get_advanced_architecture_description(iteration)
        
        changes_map = {
            6: "Transformer with Fourier features, self-attention for joint correlations",
            7: "Physics-Informed NN with differentiable FK layer, FK consistency loss",
            8: "Conditional INN for multi-solution IK, learns full solution distribution",
            9: "Denoising Diffusion model for generative IK, iterative refinement",
            10: "Mixture Density Network with uncertainty quantification"
        }
        changes = changes_map.get(iteration, "")
        
        log_iteration(iteration, arch_desc, hyperparams, metrics, changes, "Continue evaluation")
        
        pos_rmse = metrics.get('position_rmse_mm', float('inf'))
        success_rate = metrics.get('success_rate_pct', 0)
        print(f"\n  Iteration {iteration} results:")
        print(f"    Position RMSE: {pos_rmse:.4f} mm (target: < 1.0 mm)")
        print(f"    Success rate: {success_rate:.1f}% (target: > 95%)")
        
        if pos_rmse < 1.0 and success_rate > 95:
            print(f"\n  Targets met at iteration {iteration}!")
            break
    
    update_phase("advanced_training", "Advanced training complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=6, help='Model iteration (6-10)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no_physics', action='store_true')
    parser.add_argument('--no_curriculum', action='store_true')
    args = parser.parse_args()
    
    train_advanced_model(
        args.iteration,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_physics_loss=not args.no_physics,
        use_curriculum=not args.no_curriculum
    )
