import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models_advanced import (
    IKNetV6, IKNetV7, IKNetV8, IKNetV9, IKNetV10,
    DifferentiableFKLayer, create_advanced_model
)
from src.utils import Normalizer


class IKEnsemble(nn.Module):
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None,
                 device: str = 'cpu'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.device = device
        
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        self.fk_layer = DifferentiableFKLayer()
        
        self.model_types = [type(m).__name__ for m in models]
    
    def forward(self, x: torch.Tensor, strategy: str = 'weighted_avg',
                n_samples: int = 5) -> torch.Tensor:
        if strategy == 'weighted_avg':
            return self._weighted_average(x)
        elif strategy == 'best_of_n':
            return self._best_of_n(x, n_samples)
        elif strategy == 'uncertainty_weighted':
            return self._uncertainty_weighted(x)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        all_preds = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                if isinstance(model, (IKNetV8, IKNetV9)):
                    pred = model.predict_angles(x)
                elif isinstance(model, IKNetV10):
                    pred = model.predict_angles(x)
                else:
                    pred_sincos = model(x)
                    pred = self._sincos_to_angles(pred_sincos)
                
                all_preds.append(pred)
        
        all_preds = torch.stack(all_preds, dim=0)
        weights = self.weights.view(-1, 1, 1).expand(-1, batch_size, 6)
        
        sin_preds = torch.sin(all_preds)
        cos_preds = torch.cos(all_preds)
        
        avg_sin = (weights * sin_preds).sum(dim=0)
        avg_cos = (weights * cos_preds).sum(dim=0)
        
        return torch.atan2(avg_sin, avg_cos)
    
    def _best_of_n(self, x: torch.Tensor, n_samples: int = 5) -> torch.Tensor:
        batch_size = x.shape[0]
        
        all_preds = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(model, IKNetV8):
                    samples = model.sample(x, n_samples=n_samples, temperature=0.5)
                    for j in range(n_samples):
                        all_preds.append(samples[:, j, :])
                elif isinstance(model, IKNetV9):
                    for _ in range(n_samples):
                        pred = model.predict_angles(x, n_steps=50)
                        all_preds.append(pred)
                elif isinstance(model, IKNetV10):
                    samples = model.sample(x, n_samples=n_samples)
                    for j in range(n_samples):
                        all_preds.append(samples[:, j, :])
                else:
                    if isinstance(model, (IKNetV6, IKNetV7)):
                        pred_sincos = model(x)
                        pred = self._sincos_to_angles(pred_sincos)
                    else:
                        pred = model(x)
                    all_preds.append(pred)
        
        if len(all_preds) == 0:
            raise ValueError("No predictions generated")
        
        all_preds = torch.stack(all_preds, dim=0)
        n_preds = all_preds.shape[0]
        
        best_preds = torch.zeros(batch_size, 6, device=x.device)
        
        for b in range(batch_size):
            target_pose = x[b:b+1]
            best_error = float('inf')
            best_pred = all_preds[0, b]
            
            for p in range(n_preds):
                pred = all_preds[p, b:b+1]
                achieved_pose = self.fk_layer(pred)
                
                pos_error = torch.norm(achieved_pose[:, :3] - target_pose[:, :3]).item()
                
                if pos_error < best_error:
                    best_error = pos_error
                    best_pred = all_preds[p, b]
            
            best_preds[b] = best_pred
        
        return best_preds
    
    def _uncertainty_weighted(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        preds = []
        uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(model, IKNetV10):
                    pred, unc = model.predict_angles(x, return_uncertainty=True)
                    preds.append(pred)
                    uncertainties.append(unc)
                else:
                    if isinstance(model, (IKNetV6, IKNetV7)):
                        pred_sincos = model(x)
                        pred = self._sincos_to_angles(pred_sincos)
                    elif isinstance(model, (IKNetV8, IKNetV9)):
                        pred = model.predict_angles(x)
                    else:
                        pred = model(x)
                    
                    preds.append(pred)
                    uncertainties.append(torch.ones(batch_size, device=x.device) * 0.1)
        
        preds = torch.stack(preds, dim=0)
        uncertainties = torch.stack(uncertainties, dim=0)
        
        inv_unc = 1.0 / (uncertainties + 1e-6)
        weights = inv_unc / inv_unc.sum(dim=0, keepdim=True)
        weights = weights.unsqueeze(-1).expand(-1, -1, 6)
        
        sin_preds = torch.sin(preds)
        cos_preds = torch.cos(preds)
        
        avg_sin = (weights * sin_preds).sum(dim=0)
        avg_cos = (weights * cos_preds).sum(dim=0)
        
        return torch.atan2(avg_sin, avg_cos)
    
    def _sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        angles = torch.zeros(sincos.shape[0], 6, device=sincos.device)
        for i in range(6):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles
    
    def predict_angles(self, x: torch.Tensor, strategy: str = 'weighted_avg') -> torch.Tensor:
        return self.forward(x, strategy)


class CascadedIKSolver(nn.Module):
    
    def __init__(self, coarse_model: nn.Module, fine_model: nn.Module,
                 use_tto: bool = True, n_tto_steps: int = 5, device: str = 'cpu'):
        super().__init__()
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.use_tto = use_tto
        self.n_tto_steps = n_tto_steps
        self.device = device
        
        self.fk_layer = DifferentiableFKLayer()
    
    def forward(self, x: torch.Tensor, target_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.coarse_model.eval()
        with torch.no_grad():
            if isinstance(self.coarse_model, (IKNetV6, IKNetV7)):
                pred_sincos = self.coarse_model(x)
                coarse_pred = self._sincos_to_angles(pred_sincos)
            elif isinstance(self.coarse_model, IKNetV10):
                coarse_pred = self.coarse_model.predict_angles(x)
            else:
                coarse_pred = self.coarse_model(x)
        
        if isinstance(self.fine_model, IKNetV7) and hasattr(self.fine_model, 'refine_net'):
            self.fine_model.eval()
            with torch.no_grad():
                refined_pred = self.fine_model.predict_angles(x, n_refine=3)
        else:
            refined_pred = coarse_pred
        
        if self.use_tto and target_pose is not None:
            refined_pred = self._test_time_optimize(refined_pred, target_pose)
        
        return refined_pred
    
    def _test_time_optimize(self, joints: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
        joints = joints.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([joints], lr=0.01)
        
        for _ in range(self.n_tto_steps):
            optimizer.zero_grad()
            
            achieved_pose = self.fk_layer(joints)
            
            pos_loss = torch.nn.functional.mse_loss(
                achieved_pose[:, :3], target_pose[:, :3]
            )
            
            pred_ori = torch.cat([
                torch.sin(achieved_pose[:, 3:]),
                torch.cos(achieved_pose[:, 3:])
            ], dim=1)
            target_ori = torch.cat([
                torch.sin(target_pose[:, 3:]),
                torch.cos(target_pose[:, 3:])
            ], dim=1)
            ori_loss = torch.nn.functional.mse_loss(pred_ori, target_ori)
            
            loss = pos_loss + 0.5 * ori_loss
            loss.backward()
            optimizer.step()
        
        return joints.detach()
    
    def _sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        angles = torch.zeros(sincos.shape[0], 6, device=sincos.device)
        for i in range(6):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles
    
    def predict_angles(self, x: torch.Tensor, 
                       target_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, target_pose)


def create_ensemble_from_checkpoints(
    model_dir: Path,
    iterations: List[int] = [6, 7, 10],
    weights: Optional[List[float]] = None,
    device: str = 'cpu'
) -> IKEnsemble:
    model_dir = Path(model_dir)
    models = []
    
    for iteration in iterations:
        checkpoint_path = model_dir / f"best_model_iter{iteration}.pth"
        
        if not checkpoint_path.exists():
            print(f"  Warning: Checkpoint for iteration {iteration} not found")
            continue
        
        model, _ = create_advanced_model(iteration, device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
        print(f"  Loaded iteration {iteration}: {type(model).__name__}")
    
    if len(models) == 0:
        raise ValueError("No models loaded!")
    
    if weights is not None and len(weights) != len(models):
        weights = None
    
    return IKEnsemble(models, weights, device)


def create_cascaded_solver(
    model_dir: Path,
    coarse_iteration: int = 6,
    fine_iteration: int = 7,
    use_tto: bool = True,
    device: str = 'cpu'
) -> CascadedIKSolver:
    model_dir = Path(model_dir)
    
    coarse_model, _ = create_advanced_model(coarse_iteration, device)
    coarse_path = model_dir / f"best_model_iter{coarse_iteration}.pth"
    if coarse_path.exists():
        checkpoint = torch.load(coarse_path, map_location=device, weights_only=False)
        coarse_model.load_state_dict(checkpoint['model_state_dict'])
    
    fine_model, _ = create_advanced_model(fine_iteration, device)
    fine_path = model_dir / f"best_model_iter{fine_iteration}.pth"
    if fine_path.exists():
        checkpoint = torch.load(fine_path, map_location=device, weights_only=False)
        fine_model.load_state_dict(checkpoint['model_state_dict'])
    
    return CascadedIKSolver(coarse_model, fine_model, use_tto, device=device)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models"
    
    print("Creating ensemble...")
    try:
        ensemble = create_ensemble_from_checkpoints(
            model_dir, 
            iterations=[6, 7, 10],
            device='cpu'
        )
        print(f"Ensemble created with {ensemble.n_models} models")
    except Exception as e:
        print(f"Could not create ensemble: {e}")
    
    print("\nCreating cascaded solver...")
    try:
        cascaded = create_cascaded_solver(
            model_dir,
            coarse_iteration=6,
            fine_iteration=7,
            use_tto=True,
            device='cpu'
        )
        print("Cascaded solver created")
    except Exception as e:
        print(f"Could not create cascaded solver: {e}")
