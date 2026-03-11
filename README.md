# Neural Network Inverse Kinematics Solver

A neural network based inverse kinematics solver for the PUMA 560 6-DOF robotic manipulator. The project trains multiple neural network architectures to predict joint angles from target end-effector poses, replacing traditional numerical IK solvers with a faster learned approach.

## Overview

The solver uses forward kinematics data from the PUMA 560 robot model to train feedforward neural networks that map 6-DOF end-effector poses (position + orientation) to 6 joint angles. Joint limits are restricted to ensure a bijective (one-to-one) mapping between joint space and task space, which avoids the ambiguity problems inherent in general IK solutions.

Five model architectures are trained and compared iteratively:

1. **IKNetV1** - Baseline MLP with 4 hidden layers of 256 neurons
2. **IKNetV2** - Deeper network with BatchNorm, 5 layers from 512 down to 128
3. **IKNetV3** - Residual connections for better gradient flow
4. **IKNetV4** - Sin/cos output encoding to handle angle wrapping
5. **IKNetV5** - Multi-head architecture with separate position and orientation heads

## Project Structure

```
.
├── src/
│   ├── robot_model.py      # PUMA 560 FK/IK using roboticstoolbox
│   ├── data_generator.py   # Training data generation (uniform, singularity, boundary)
│   ├── dataset.py          # PyTorch Dataset and DataLoader
│   ├── model.py            # Neural network architectures (V1-V5)
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Evaluation metrics and benchmarking
│   ├── ik_solver.py        # Production solver API
│   ├── visualization.py    # Plot generation
│   ├── trajectory.py       # Trajectory generation (circle, helix, line)
│   └── utils.py            # Normalization, logging, helpers
├── scripts/
│   ├── run_all.py          # Full pipeline execution
│   ├── run_web.py          # Launch web dashboard
│   └── debug_data.py       # Data inspection utility
├── web/
│   ├── app.py              # Flask web application
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS, generated plots
├── data/                   # Generated training/test datasets (.npz)
├── models/                 # Saved model checkpoints (.pth)
├── results/                # Evaluation metrics and error data
└── requirements.txt        # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

This generates training data, trains all 5 model iterations, evaluates them, and produces visualizations:

```bash
python scripts/run_all.py
```

The pipeline has 5 phases:

1. Robot model verification
2. Data generation (uniform, singularity, and boundary sampling)
3. Iterative training of all 5 architectures
4. Full evaluation and numerical IK benchmarking
5. Visualization generation

Progress is tracked in `context_log.json`, so the pipeline can resume from where it left off if interrupted.

### Launch the Web Dashboard

```bash
python scripts/run_web.py
```

Then open `http://localhost:5000` in your browser. The dashboard provides:

- Metrics overview across all training iterations
- Live IK prediction with 3D arm visualization
- Trajectory planning and animation
- Error distribution plots and training curves

### Run Individual Components

Generate training data only:

```bash
python src/data_generator.py
```

Train models only:

```bash
python src/train.py
```

Generate visualizations only:

```bash
python src/visualization.py
```

## Data Generation

Training data is created by sampling joint configurations and computing forward kinematics to get corresponding end-effector poses. Three sampling strategies are used:

- **Uniform**: 100,000 samples across the full restricted joint space
- **Singularity**: 50,000 samples concentrated near singular configurations
- **Boundary**: 25,000 samples near joint limits

The combined dataset is normalized and split into 70% training, 15% validation, and 15% test sets.

## Evaluation Metrics

Models are evaluated on:

- Position RMSE in millimeters
- Orientation RMSE in degrees
- Per-joint prediction error
- Inference time compared to numerical Levenberg-Marquardt IK

## Dependencies

- PyTorch (neural network training and inference)
- roboticstoolbox-python (PUMA 560 model and forward kinematics)
- spatialmath-python (SE3 transformations)
- NumPy, SciPy, Matplotlib (numerical computing and plotting)
- Flask (web dashboard)
- tqdm (progress bars)

## License

This project is provided as-is for educational and research purposes.
