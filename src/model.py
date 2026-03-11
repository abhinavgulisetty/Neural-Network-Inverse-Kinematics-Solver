"""
Neural Network architectures for Inverse Kinematics solving.
Multiple architectures for iterative experimentation.
"""
import torch
import torch.nn as nn


class IKNetV1(nn.Module):
    """Iteration 1: Baseline MLP with 4 hidden layers, 256 neurons."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.net(x)


class IKNetV2(nn.Module):
    """Iteration 2: Deeper network with BatchNorm, 5 layers, wider, with batch normalization."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with optional projection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.project = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.project(x))


class IKNetV3(nn.Module):
    """Iteration 3: Residual connections for better gradient flow."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(6, 256), nn.ReLU())
        self.res1 = ResidualBlock(256, 256)
        self.res2 = ResidualBlock(256, 256)
        self.res3 = ResidualBlock(256, 128)
        self.res4 = ResidualBlock(128, 128)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(128, 6)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(self.res1(x))
        x = self.dropout(self.res2(x))
        x = self.dropout(self.res3(x))
        x = self.dropout(self.res4(x))
        return self.output(x)


class IKNetV4(nn.Module):
    """Iteration 4: Sin/Cos output encoding to handle angle wrapping."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 12)
        )

    def forward(self, x):
        out = self.backbone(x)
        return out

    def predict_angles(self, x):
        """Convert sin/cos outputs to angles using atan2."""
        out = self.forward(x)
        angles = torch.zeros(x.shape[0], 6, device=x.device)
        for i in range(6):
            sin_val = out[:, 2 * i]
            cos_val = out[:, 2 * i + 1]
            angles[:, i] = torch.atan2(sin_val, cos_val)
        return angles


class IKNetV5(nn.Module):
    """Iteration 5: Multi-head with separate heads for position joints (1-3) and wrist joints (4-6)."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.ori_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        shared = self.backbone(x)
        pos_joints = self.pos_head(shared)
        ori_joints = self.ori_head(shared)
        return torch.cat([pos_joints, ori_joints], dim=1)


MODEL_REGISTRY = {
    1: ("IKNetV1 - Baseline MLP (4x256)", IKNetV1),
    2: ("IKNetV2 - Deeper + BatchNorm (512 to 128)", IKNetV2),
    3: ("IKNetV3 - Residual Connections", IKNetV3),
    4: ("IKNetV4 - Sin/Cos Output Encoding", IKNetV4),
    5: ("IKNetV5 - Multi-Head (Position + Orientation)", IKNetV5),
}


def create_model(iteration):
    """Create model by iteration number."""
    if iteration not in MODEL_REGISTRY:
        raise ValueError(f"Unknown iteration {iteration}. Available: {list(MODEL_REGISTRY.keys())}")
    name, cls = MODEL_REGISTRY[iteration]
    print(f"\n  Creating model: {name}")
    model = cls()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model, name


def get_architecture_description(iteration):
    """Get human-readable architecture description."""
    if iteration in MODEL_REGISTRY:
        return MODEL_REGISTRY[iteration][0]
    return "Unknown"
