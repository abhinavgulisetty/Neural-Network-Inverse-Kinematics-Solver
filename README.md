# SOTA Neural Network Inverse Kinematics Solver

A state-of-the-art neural network based inverse kinematics solver for the PUMA 560 6-DOF robotic manipulator. This project implements cutting-edge deep learning architectures including Transformers, Physics-Informed Neural Networks (PINNs), Conditional Invertible Neural Networks (cINNs), Diffusion Models, and Mixture Density Networks (MDNs) for learning the inverse kinematics mapping.

## Key Features

- **10 Neural Network Architectures** (V1-V10): From baseline MLPs to SOTA architectures
- **Physics-Informed Learning**: Differentiable FK layer for physics-based loss functions
- **Multi-Solution Handling**: cINN and Diffusion models can sample multiple valid IK solutions
- **Uncertainty Quantification**: MDN provides confidence estimates for predictions
- **Test-Time Optimization**: Gradient-based refinement for sub-millimeter accuracy
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Curriculum Learning**: Progressive training from easy to hard samples
- **GPU Acceleration**: Mixed precision training with automatic device selection

## Model Architectures

### Basic Models (V1-V5)
| Version | Architecture | Description |
|---------|-------------|-------------|
| V1 | Baseline MLP | 4 hidden layers, 256 neurons, ReLU, Dropout |
| V2 | Deep + BatchNorm | 5 layers (512→128), BatchNorm, improved stability |
| V3 | Residual Network | Skip connections for gradient flow |
| V4 | Sin/Cos Encoding | Handles angle discontinuities at ±π |
| V5 | Multi-Head | Separate position (J1-3) and orientation (J4-6) heads |

### Advanced Models (V6-V10)
| Version | Architecture | Key Innovation |
|---------|-------------|----------------|
| V6 | **Transformer** | Self-attention for joint correlations, Fourier positional encoding |
| V7 | **Physics-Informed (PINN)** | Differentiable FK layer, FK consistency loss, iterative refinement |
| V8 | **Conditional INN (cINN)** | Invertible architecture, learns full IK solution distribution |
| V9 | **Diffusion Model** | Denoising diffusion, iterative sampling, multi-modal solutions |
| V10 | **Mixture Density Network** | Gaussian mixture output, uncertainty quantification |

## Project Structure

```
.
├── src/
│   ├── robot_model.py          # PUMA 560 FK/IK using roboticstoolbox
│   ├── data_generator.py       # Training data generation
│   ├── dataset.py              # PyTorch Dataset and DataLoader
│   ├── model.py                # Basic architectures (V1-V5) + registry
│   ├── models_advanced.py      # SOTA architectures (V6-V10)
│   ├── train.py                # Basic training loop
│   ├── train_advanced.py       # Advanced training (physics loss, curriculum)
│   ├── evaluate.py             # Basic evaluation
│   ├── evaluate_advanced.py    # Comprehensive benchmarking
│   ├── ensemble.py             # Ensemble methods
│   ├── ik_solver.py            # Production solver API
│   ├── visualization.py        # Plot generation
│   ├── trajectory.py           # Trajectory generation
│   └── utils.py                # Normalization, logging, helpers
├── scripts/
│   ├── run_all.py              # Basic pipeline (V1-V5)
│   ├── run_advanced.py         # Advanced pipeline (V6-V10)
│   ├── run_web.py              # Launch web dashboard
│   └── debug_data.py           # Data inspection
├── web/
│   ├── app.py                  # Flask web application
│   ├── templates/              # HTML templates
│   └── static/                 # CSS, JS, plots
├── data/                       # Training/test datasets (.npz)
├── models/                     # Model checkpoints (.pth)
├── results/                    # Evaluation metrics
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Basic Pipeline (V1-V5)
```bash
python scripts/run_all.py
```

### 2. Run Advanced Pipeline (V6-V10)
```bash
# Full pipeline: data check, train all, evaluate, ensemble
python scripts/run_advanced.py --all

# Train specific models
python scripts/run_advanced.py --train 6 7 10

# Evaluate trained models
python scripts/run_advanced.py --evaluate

# Test ensemble methods
python scripts/run_advanced.py --ensemble

# View project summary
python scripts/run_advanced.py --summary
```

### 3. Launch Web Dashboard
```bash
python scripts/run_web.py
# Open http://localhost:5000
```

## Training Options

```bash
python scripts/run_advanced.py --train 6 7 8 9 10 \
    --epochs 200 \
    --batch_size 256 \
    --no_tto  # Disable test-time optimization
```

### Training Features
- **Curriculum Learning**: Starts with easy samples (near home position), gradually introduces harder ones
- **Physics-Informed Loss**: MSE on joints + FK consistency + joint limit penalty
- **Warmup + Cosine Annealing**: Learning rate schedule for stable training
- **Early Stopping**: Patience-based with best model restoration
- **Mixed Precision**: FP16 training on GPU for 2x speedup

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Position RMSE | End-effector position error | < 1 mm |
| Orientation RMSE | Euler angle error | < 0.5° |
| Success Rate (1mm) | % samples with pos < 1mm AND ori < 0.5° | > 95% |
| Success Rate (5mm) | Relaxed criteria | > 99% |
| Inference Time | Single sample prediction | < 1 ms |

## Advanced Features

### Multi-Solution Sampling (V8, V9, V10)
```python
from src.models_advanced import IKNetV8

model = IKNetV8()
# Sample 10 different IK solutions for one pose
solutions = model.sample(target_pose, n_samples=10, temperature=1.0)
```

### Uncertainty Quantification (V10)
```python
from src.models_advanced import IKNetV10

model = IKNetV10()
joints, uncertainty = model.predict_angles(pose, return_uncertainty=True)
print(f"Predicted joints: {joints}, Uncertainty: {uncertainty:.3f}")
```

### Ensemble Prediction
```python
from src.ensemble import create_ensemble_from_checkpoints

ensemble = create_ensemble_from_checkpoints(
    model_dir="models/",
    iterations=[6, 7, 10]
)
# Strategies: 'weighted_avg', 'best_of_n', 'uncertainty_weighted'
joints = ensemble.predict_angles(pose, strategy='best_of_n')
```

### Test-Time Optimization
```python
from src.train_advanced import TestTimeOptimizer

tto = TestTimeOptimizer(robot_model, n_steps=10, lr=0.01)
refined_joints = tto.refine(initial_joints, target_pose)
```

## Technical Details

### Data Generation
- **Uniform**: 100K samples across joint space
- **Singularity**: 50K samples near singular configurations
- **Boundary**: 25K samples near joint limits
- **Split**: 70% train, 15% validation, 15% test

### PUMA 560 Joint Limits (Restricted for Bijective IK)
| Joint | Lower | Upper |
|-------|-------|-------|
| 1 | -90° | +90° |
| 2 | -90° | 0° |
| 3 | 0° | +90° |
| 4 | -90° | +90° |
| 5 | -90° | +90° |
| 6 | -90° | +90° |

### Loss Functions
- **Basic**: MSE on normalized joint angles
- **Sin/Cos**: MSE on sin/cos representation (V4, V6, V7)
- **Physics-Informed**: `L_total = L_joint + λ_fk * L_FK + λ_limit * L_limits`
- **cINN**: Negative log-likelihood under normalizing flow
- **Diffusion**: MSE on noise prediction
- **MDN**: Negative log-likelihood under Gaussian mixture

## Results

Expected performance after training (depends on hyperparameters and epochs):

| Model | Pos RMSE (mm) | Success 5mm | Inference (ms) |
|-------|---------------|-------------|----------------|
| V4 (Sin/Cos) | ~15-20 | ~60-70% | ~0.3 |
| V6 (Transformer) | ~5-10 | ~85-90% | ~0.5 |
| V7 (PINN) | ~3-8 | ~90-95% | ~0.8 |
| V8 (cINN) | ~5-10 | ~85-90% | ~1.0 |
| V10 (MDN) | ~5-12 | ~80-90% | ~0.4 |
| V7 + TTO | ~0.5-2 | ~98-99% | ~5-10 |
| Ensemble | ~2-5 | ~95-98% | ~2-5 |

## Research References

- **Transformers**: Vaswani et al., "Attention Is All You Need" (2017)
- **Fourier Features**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions" (2020)
- **Physics-Informed NN**: Raissi et al., "Physics-informed neural networks" (2019)
- **Normalizing Flows**: Ardizzone et al., "Guided Image Generation with Conditional Invertible Neural Networks" (2019)
- **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **Mixture Density Networks**: Bishop, "Mixture Density Networks" (1994)

## Dependencies

Core:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- roboticstoolbox-python >= 1.1.0
- spatialmath-python >= 1.1.0

Visualization:
- Matplotlib >= 3.7.0
- Plotly >= 5.15.0
- Seaborn >= 0.12.0

Web:
- Flask >= 3.0.0

## License

This project is provided for educational and research purposes.

## Citation

If you use this code in your research, please cite:
```
@software{sota_ik_solver,
  title={SOTA Neural Network Inverse Kinematics Solver},
  year={2024},
  description={State-of-the-art deep learning architectures for robotic inverse kinematics}
}
```
