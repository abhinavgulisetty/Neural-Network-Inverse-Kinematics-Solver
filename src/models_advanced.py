import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class FourierFeatures(nn.Module):
    
    def __init__(self, input_dim: int, num_frequencies: int = 64, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        B = torch.randn(input_dim, num_frequencies) * scale
        self.register_buffer('B', B)
        self.output_dim = input_dim + 2 * num_frequencies
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = self.B.to(x.dtype)
        x_proj = 2 * math.pi * x @ B
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SE3PoseEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.output_dim = 3 + 9 + 6
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        batch_size = pose.shape[0]
        pos = pose[:, :3]
        rpy = pose[:, 3:]
        
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        
        R = torch.zeros(batch_size, 9, device=pose.device, dtype=pose.dtype)
        R[:, 0] = cy * cp
        R[:, 1] = cy * sp * sr - sy * cr
        R[:, 2] = cy * sp * cr + sy * sr
        R[:, 3] = sy * cp
        R[:, 4] = sy * sp * sr + cy * cr
        R[:, 5] = sy * sp * cr - cy * sr
        R[:, 6] = -sp
        R[:, 7] = cp * sr
        R[:, 8] = cp * cr
        
        pos_norm = torch.norm(pos, dim=1, keepdim=True)
        sin_rpy = torch.sin(rpy)
        cos_rpy = torch.cos(rpy)
        
        additional = torch.cat([sin_rpy, cos_rpy], dim=1)
        
        return torch.cat([pos, R, additional], dim=1)


class SinCosJointEncoder(nn.Module):
    
    def __init__(self, n_joints: int = 6):
        super().__init__()
        self.n_joints = n_joints
        self.output_dim = 2 * n_joints
    
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        sincos = torch.zeros(angles.shape[0], self.output_dim, device=angles.device)
        for i in range(self.n_joints):
            sincos[:, 2*i] = torch.sin(angles[:, i])
            sincos[:, 2*i + 1] = torch.cos(angles[:, i])
        return sincos
    
    def decode(self, sincos: torch.Tensor) -> torch.Tensor:
        angles = torch.zeros(sincos.shape[0], self.n_joints, device=sincos.device)
        for i in range(self.n_joints):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles


class GatedLinearUnit(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class TransformerBlock(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        
        return x


class IKNetV6(nn.Module):
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 6, 
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.fourier = FourierFeatures(6, num_frequencies=64, scale=10.0)
        self.input_proj = nn.Linear(self.fourier.output_dim, d_model)
        
        self.joint_queries = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        
        self.joint_pos_embed = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.joint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Linear(d_model // 2, 2)
            )
            for _ in range(6)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x_encoded = self.fourier(x)
        pose_token = self.input_proj(x_encoded).unsqueeze(1)
        
        joint_queries = self.joint_queries.expand(batch_size, -1, -1)
        joint_queries = joint_queries + self.joint_pos_embed
        
        tokens = torch.cat([pose_token, joint_queries], dim=1)
        
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        tokens = self.norm(tokens)
        
        joint_tokens = tokens[:, 1:, :]
        
        outputs = []
        for i in range(6):
            joint_out = self.joint_heads[i](joint_tokens[:, i, :])
            outputs.append(joint_out)
        
        return torch.cat(outputs, dim=1)
    
    def predict_angles(self, x: torch.Tensor) -> torch.Tensor:
        sincos = self.forward(x)
        angles = torch.zeros(x.shape[0], 6, device=x.device)
        for i in range(6):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles


class DifferentiableFKLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.register_buffer('dh_params', torch.tensor([
            [0.0,      -np.pi/2,  0.0,     0.0],
            [0.4318,   0.0,       0.0,     0.0],
            [0.0203,   -np.pi/2,  0.0,     0.0],
            [0.0,      np.pi/2,   0.4318,  0.0],
            [0.0,      -np.pi/2,  0.0,     0.0],
            [0.0,      0.0,       0.0,     0.0],
        ], dtype=torch.float32))
    
    def dh_matrix(self, theta: torch.Tensor, a: float, alpha: float, d: float) -> torch.Tensor:
        batch_size = theta.shape[0]
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        
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
    
    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        batch_size = joint_angles.shape[0]
        
        T = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype)
        T = T.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        for i in range(6):
            a = self.dh_params[i, 0].item()
            alpha = self.dh_params[i, 1].item()
            d = self.dh_params[i, 2].item()
            Ti = self.dh_matrix(joint_angles[:, i], a, alpha, d)
            T = torch.bmm(T, Ti)
        
        pos = T[:, :3, 3]
        
        R = T[:, :3, :3]
        
        pitch = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2))
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        
        rpy = torch.stack([roll, pitch, yaw], dim=1)
        
        return torch.cat([pos, rpy], dim=1)


class IKNetV7(nn.Module):
    
    def __init__(self, hidden_dim: int = 512, n_layers: int = 6):
        super().__init__()
        
        self.pose_encoder = SE3PoseEncoder()
        
        layers = []
        input_dim = self.pose_encoder.output_dim
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(n_layers - 1):
            layers.append(ResidualMLPBlock(hidden_dim, hidden_dim))
        
        self.backbone = nn.Sequential(*layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 12)
        )
        
        self.fk_layer = DifferentiableFKLayer()
        
        self.refine_net = nn.Sequential(
            nn.Linear(12 + 6, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 12)
        )
    
    def forward(self, x: torch.Tensor, n_refine: int = 0) -> torch.Tensor:
        x_encoded = self.pose_encoder(x)
        
        features = self.backbone(x_encoded)
        sincos = self.output_head(features)
        
        if n_refine > 0:
            for _ in range(n_refine):
                angles = self.sincos_to_angles(sincos)
                
                achieved_pose = self.fk_layer(angles)
                
                pose_error = x - achieved_pose
                
                refine_input = torch.cat([sincos, pose_error], dim=1)
                delta = self.refine_net(refine_input)
                sincos = sincos + delta
        
        return sincos
    
    def sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        angles = torch.zeros(sincos.shape[0], 6, device=sincos.device)
        for i in range(6):
            angles[:, i] = torch.atan2(sincos[:, 2*i], sincos[:, 2*i + 1])
        return angles
    
    def predict_angles(self, x: torch.Tensor, n_refine: int = 0) -> torch.Tensor:
        sincos = self.forward(x, n_refine)
        return self.sincos_to_angles(sincos)
    
    def compute_fk(self, joint_angles: torch.Tensor) -> torch.Tensor:
        return self.fk_layer(joint_angles)


class ResidualMLPBlock(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CouplingLayer(nn.Module):
    
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        
        self.s_net = nn.Sequential(
            nn.Linear(self.split_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.split_dim),
            nn.Tanh()
        )
        
        self.t_net = nn.Sequential(
            nn.Linear(self.split_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.split_dim)
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        if not reverse:
            s = self.s_net(torch.cat([x1, c], dim=1)) * 2
            t = self.t_net(torch.cat([x1, c], dim=1))
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            s = self.s_net(torch.cat([x1, c], dim=1)) * 2
            t = self.t_net(torch.cat([x1, c], dim=1))
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
        
        return torch.cat([y1, y2], dim=1), log_det


class IKNetV8(nn.Module):
    
    def __init__(self, n_joints: int = 6, cond_dim: int = 6, n_coupling_layers: int = 8, 
                 hidden_dim: int = 256):
        super().__init__()
        
        self.n_joints = n_joints
        self.latent_dim = n_joints
        
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_dim = hidden_dim
        
        self.coupling_layers = nn.ModuleList()
        for i in range(n_coupling_layers):
            self.coupling_layers.append(
                CouplingLayer(n_joints, self.cond_dim, hidden_dim)
            )
        
        perms = []
        for _ in range(n_coupling_layers):
            perm = torch.randperm(n_joints)
            perms.append(perm)
        self.register_buffer('permutations', torch.stack(perms))
        
    def forward(self, joints: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.cond_encoder(pose)
        z = joints
        log_det_total = torch.zeros(joints.shape[0], device=joints.device)
        
        for i, layer in enumerate(self.coupling_layers):
            z, log_det = layer(z, c, reverse=False)
            log_det_total = log_det_total + log_det
            z = z[:, self.permutations[i]]
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        c = self.cond_encoder(pose)
        x = z
        
        for i in range(len(self.coupling_layers) - 1, -1, -1):
            inv_perm = torch.argsort(self.permutations[i])
            x = x[:, inv_perm]
            x, _ = self.coupling_layers[i](x, c, reverse=True)
        
        return x
    
    def sample(self, pose: torch.Tensor, n_samples: int = 1, 
               temperature: float = 1.0) -> torch.Tensor:
        batch_size = pose.shape[0]
        device = pose.device
        
        all_samples = []
        for _ in range(n_samples):
            z = torch.randn(batch_size, self.latent_dim, device=device) * temperature
            joints = self.inverse(z, pose)
            all_samples.append(joints)
        
        return torch.stack(all_samples, dim=1)
    
    def predict_angles(self, pose: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(pose.shape[0], self.latent_dim, device=pose.device)
        return self.inverse(z, pose)
    
    def log_likelihood(self, joints: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(joints, pose)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return log_pz + log_det


class SinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class DiffusionBlock(nn.Module):
    
    def __init__(self, dim: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.time_mlp = nn.Linear(time_dim, dim)
        self.cond_mlp = nn.Linear(cond_dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = h + self.time_mlp(t_emb) + self.cond_mlp(c)
        return x + self.net(h)


class IKNetV9(nn.Module):
    
    def __init__(self, n_joints: int = 6, hidden_dim: int = 512, n_layers: int = 8,
                 n_timesteps: int = 1000):
        super().__init__()
        
        self.n_joints = n_joints
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim
        
        time_dim = hidden_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        self.cond_encoder = nn.Sequential(
            FourierFeatures(6, num_frequencies=32, scale=10.0),
            nn.Linear(6 + 64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(n_joints, hidden_dim)
        
        self.blocks = nn.ModuleList([
            DiffusionBlock(hidden_dim, time_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_joints)
        )
        
        self._setup_noise_schedule()
        
    def _setup_noise_schedule(self):
        steps = self.n_timesteps
        s = 0.008
        
        t = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())
        c = self.cond_encoder(pose)
        
        h = self.input_proj(x_noisy)
        
        for block in self.blocks:
            h = block(h, t_emb, c)
        
        return self.output_proj(h)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        x_noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return x_noisy, noise
    
    @torch.no_grad()
    def sample(self, pose: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.n_timesteps
        
        batch_size = pose.shape[0]
        device = pose.device
        
        x = torch.randn(batch_size, self.n_joints, device=device)
        
        for i in range(n_steps - 1, -1, -1):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            noise_pred = self.forward(x, t, pose)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            )
            
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise
        
        return x
    
    def predict_angles(self, pose: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        return self.sample(pose, n_steps)


class IKNetV10(nn.Module):
    
    def __init__(self, n_joints: int = 6, hidden_dim: int = 512, n_components: int = 5):
        super().__init__()
        
        self.n_joints = n_joints
        self.n_components = n_components
        
        self.fourier = FourierFeatures(6, num_frequencies=48, scale=10.0)
        
        self.backbone = nn.Sequential(
            nn.Linear(self.fourier.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            ResidualMLPBlock(hidden_dim, hidden_dim),
            ResidualMLPBlock(hidden_dim, hidden_dim),
            ResidualMLPBlock(hidden_dim, hidden_dim),
            ResidualMLPBlock(hidden_dim, hidden_dim),
            
            nn.LayerNorm(hidden_dim),
        )
        
        self.weight_head = nn.Linear(hidden_dim, n_components)
        
        self.mean_head = nn.Linear(hidden_dim, n_components * n_joints * 2)
        
        self.logvar_head = nn.Linear(hidden_dim, n_components * n_joints)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_enc = self.fourier(x)
        features = self.backbone(x_enc)
        
        weights = F.softmax(self.weight_head(features), dim=-1)
        means_flat = self.mean_head(features)
        logvars_flat = self.logvar_head(features)
        
        batch_size = x.shape[0]
        means = means_flat.view(batch_size, self.n_components, self.n_joints, 2)
        logvars = logvars_flat.view(batch_size, self.n_components, self.n_joints)
        
        return {
            'weights': weights,
            'means': means,
            'logvars': logvars
        }
    
    def predict_angles(self, x: torch.Tensor, return_uncertainty: bool = False) -> torch.Tensor:
        out = self.forward(x)
        weights = out['weights']
        means = out['means']
        logvars = out['logvars']
        
        best_idx = weights.argmax(dim=1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        
        best_means = means[batch_idx, best_idx]
        
        angles = torch.atan2(best_means[:, :, 0], best_means[:, :, 1])
        
        if return_uncertainty:
            best_vars = torch.exp(logvars[batch_idx, best_idx])
            uncertainty = torch.sqrt(best_vars.mean(dim=1))
            return angles, uncertainty
        
        return angles
    
    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        out = self.forward(x)
        weights = out['weights']
        means = out['means']
        logvars = out['logvars']
        
        batch_size = x.shape[0]
        device = x.device
        
        samples = []
        for _ in range(n_samples):
            comp_idx = torch.multinomial(weights, 1).squeeze(-1)
            batch_idx = torch.arange(batch_size, device=device)
            
            comp_means = means[batch_idx, comp_idx]
            comp_vars = torch.exp(logvars[batch_idx, comp_idx])
            
            base_angles = torch.atan2(comp_means[:, :, 0], comp_means[:, :, 1])
            
            noise = torch.randn_like(base_angles) * torch.sqrt(comp_vars)
            sample = base_angles + noise
            
            samples.append(sample)
        
        return torch.stack(samples, dim=1)
    
    def nll_loss(self, x: torch.Tensor, target_angles: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        weights = out['weights']
        means = out['means']
        logvars = out['logvars']
        
        batch_size = x.shape[0]
        
        target_sincos = torch.stack([
            torch.sin(target_angles),
            torch.cos(target_angles)
        ], dim=-1)
        
        log_probs = []
        for k in range(self.n_components):
            comp_mean = means[:, k]
            comp_logvar = logvars[:, k]
            comp_var = torch.exp(comp_logvar)
            
            diff = target_sincos - comp_mean
            diff_sq = (diff ** 2).sum(dim=-1)
            
            log_prob = -0.5 * (diff_sq / comp_var + comp_logvar + math.log(2 * math.pi))
            log_prob = log_prob.sum(dim=-1)
            
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)
        
        log_weights = torch.log(weights + 1e-10)
        log_prob_mixture = torch.logsumexp(log_weights + log_probs, dim=1)
        
        return -log_prob_mixture.mean()


ADVANCED_MODEL_REGISTRY = {
    6: ("IKNetV6 - Transformer with Fourier Features", IKNetV6),
    7: ("IKNetV7 - Physics-Informed (PINN) with FK Loss", IKNetV7),
    8: ("IKNetV8 - Conditional INN (Multi-Solution)", IKNetV8),
    9: ("IKNetV9 - Diffusion Model", IKNetV9),
    10: ("IKNetV10 - Mixture Density Network (MDN)", IKNetV10),
}


def create_advanced_model(iteration: int, device: str = 'cpu') -> Tuple[nn.Module, str]:
    if iteration not in ADVANCED_MODEL_REGISTRY:
        raise ValueError(f"Unknown iteration {iteration}. Available: {list(ADVANCED_MODEL_REGISTRY.keys())}")
    
    name, cls = ADVANCED_MODEL_REGISTRY[iteration]
    print(f"\n  Creating model: {name}")
    
    model = cls()
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    return model, name


def get_advanced_architecture_description(iteration: int) -> str:
    if iteration in ADVANCED_MODEL_REGISTRY:
        return ADVANCED_MODEL_REGISTRY[iteration][0]
    return "Unknown"
