import torch
import torch.nn as nn


class IKNetV1(nn.Module):

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
        out = self.forward(x)
        angles = torch.zeros(x.shape[0], 6, device=x.device)
        for i in range(6):
            sin_val = out[:, 2 * i]
            cos_val = out[:, 2 * i + 1]
            angles[:, i] = torch.atan2(sin_val, cos_val)
        return angles


class IKNetV5(nn.Module):

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


def _get_extended_registry():
    from src.models_advanced import ADVANCED_MODEL_REGISTRY
    extended = MODEL_REGISTRY.copy()
    extended.update(ADVANCED_MODEL_REGISTRY)
    return extended


def create_model(iteration, device='cpu'):
    if iteration <= 5:
        if iteration not in MODEL_REGISTRY:
            raise ValueError(f"Unknown iteration {iteration}. Available: 1-10")
        name, cls = MODEL_REGISTRY[iteration]
        print(f"\n  Creating model: {name}")
        model = cls()
    else:
        from src.models_advanced import create_advanced_model
        model, name = create_advanced_model(iteration, device)
        return model, name
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model, name


def get_architecture_description(iteration):
    if iteration <= 5:
        if iteration in MODEL_REGISTRY:
            return MODEL_REGISTRY[iteration][0]
    else:
        from src.models_advanced import get_advanced_architecture_description
        return get_advanced_architecture_description(iteration)
    return "Unknown"


def list_all_models():
    print("\n=== Available IK Model Architectures ===\n")
    print("Basic Models (V1-V5):")
    for i, (name, _) in MODEL_REGISTRY.items():
        print(f"  {i}: {name}")
    
    print("\nAdvanced Models (V6-V10):")
    try:
        from src.models_advanced import ADVANCED_MODEL_REGISTRY
        for i, (name, _) in ADVANCED_MODEL_REGISTRY.items():
            print(f"  {i}: {name}")
    except ImportError:
        print("  (Advanced models not available)")
    
    print()
    return list(MODEL_REGISTRY.keys()) + [6, 7, 8, 9, 10]
