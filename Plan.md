# Plan.md — Neural Network-Based Inverse Kinematics Solver for 6-DOF Manipulator

> **For: Claude Code** — Execute this plan top-to-bottom. Each phase builds on the previous one.
> **Platform**: Linux, CPU-only training (no CUDA). Use Python, PyTorch (CPU), and a simple web dashboard.
> **Workspace**: `/home/zypher/Music/Shi go down/`

---

## 🔑 Context Persistence System (READ FIRST)

Before doing ANY work, check if `context_log.json` exists in the project root.
- If it **exists**, read it. It contains a JSON log of every completed phase, iteration results, model metrics, architecture changes, and decisions made so far. Resume from where it left off.
- If it **doesn't exist**, create it with an empty structure and start from Phase 1.

**After EVERY phase/sub-phase**, update `context_log.json` with:
```json
{
  "last_completed_phase": "phase_id",
  "timestamp": "ISO-8601",
  "iterations": [
    {
      "iteration": 1,
      "architecture": "description",
      "hyperparameters": {},
      "metrics": {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "position_rmse_mm": 0.0,
        "orientation_rmse_deg": 0.0,
        "success_rate_pct": 0.0,
        "avg_inference_ms": 0.0
      },
      "changes_made": "description of what changed",
      "next_steps": "what to try next"
    }
  ],
  "best_model": {
    "iteration": 0,
    "path": "models/best_model.pth",
    "metrics": {}
  },
  "decisions": ["list of key decisions and why"]
}
```

This is **critical** — if your context window runs out mid-work, the next invocation reads this file and continues seamlessly.

---

## Project Directory Structure

Create this structure at the start:

```
/home/zypher/Music/Shi go down/
├── Plan.md                    # This file
├── document.md                # Original project spec (already exists)
├── context_log.json           # Persistent context across iterations
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── robot_model.py         # DH parameters, FK engine, joint limits
│   ├── data_generator.py      # Training data generation (FK → dataset)
│   ├── dataset.py             # PyTorch Dataset/DataLoader classes
│   ├── model.py               # Neural network architecture(s)
│   ├── train.py               # Training loop with early stopping
│   ├── evaluate.py            # Evaluation: metrics, comparisons
│   ├── ik_solver.py           # Production IK solver wrapper
│   ├── trajectory.py          # Trajectory generation (circle, helix, line)
│   ├── visualization.py       # Matplotlib plots for arm movement, errors
│   └── utils.py               # Shared utilities, normalization, context_log I/O
├── models/                    # Saved model checkpoints
│   └── .gitkeep
├── data/                      # Generated datasets
│   └── .gitkeep
├── results/                   # Evaluation results, plots, CSVs
│   └── .gitkeep
├── web/                       # Web dashboard
│   ├── app.py                 # Flask web server
│   ├── templates/
│   │   └── index.html         # Main dashboard page
│   └── static/
│       ├── css/
│       │   └── style.css
│       ├── js/
│       │   └── main.js
│       └── plots/             # Generated plot images served to frontend
└── scripts/
    ├── run_all.py             # Master script: runs everything end-to-end
    └── run_web.py             # Starts the web server
```

---

## Phase 0: Environment Setup

1. Create `requirements.txt`:
   ```
   torch>=2.0.0
   numpy>=1.24.0
   scipy>=1.10.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   pandas>=2.0.0
   flask>=3.0.0
   plotly>=5.15.0
   scikit-learn>=1.3.0
   tqdm>=4.65.0
   roboticstoolbox-python>=1.1.0
   spatialmath-python>=1.1.0
   ```

2. Run: `pip install -r requirements.txt`

3. Create `context_log.json` with initial empty structure.

---

## Phase 1: Robot Model & Forward Kinematics Engine

### File: `src/robot_model.py`

Implement a class `RobotModel` that:

1. **Defines the PUMA 560 robot** using Denavit-Hartenberg (DH) parameters from Peter Corke's Robotics Toolbox:
   ```python
   # PUMA 560 DH Parameters (standard convention)
   # Link | a (m)   | alpha (rad) | d (m)   | theta offset
   # 1    | 0       | π/2         | 0       | 0
   # 2    | 0.4318  | 0           | 0       | 0
   # 3    | 0.0203  | -π/2        | 0.15005 | 0
   # 4    | 0       | π/2         | 0.4318  | 0
   # 5    | 0       | -π/2        | 0       | 0
   # 6    | 0       | 0           | 0       | 0
   ```

2. **Joint limits** (in radians): Define per-joint min/max limits (typically ±2.79 to ±5.02 depending on joint).

3. **Forward kinematics** method `forward_kinematics(joint_angles) → (position_xyz, orientation_euler_zyx)`:
   - Takes 6 joint angles.
   - Returns 3D position (x, y, z) in meters and orientation as Euler ZYX angles (3 values) in radians.
   - Use `roboticstoolbox` library's `DHRobot` for reliable FK computation.

4. **Batch forward kinematics** method for generating large datasets efficiently.

5. **Numerical IK baseline** using `roboticstoolbox`'s built-in IK solver for comparison.

6. **Visualization helper**: method to return joint positions for all links (for 3D arm plotting).

### Verification:
- Test FK by computing forward kinematics for known joint configurations (e.g., home position `[0,0,0,0,0,0]`) and verify the end-effector position matches expected values.
- Test round-trip: FK → numerical IK → FK should return the original pose within tolerance.

---

## Phase 2: Training Data Generation

### File: `src/data_generator.py`

Implement `DataGenerator` class:

1. **Phase 2a — Uniform Sampling (100,000 samples)**:
   - Sample joint angles uniformly within joint limits.
   - Compute FK for each sample → get (position, orientation).
   - Store as: `inputs = [x, y, z, roll, pitch, yaw]` (6D), `targets = [θ1, θ2, θ3, θ4, θ5, θ6]` (6D).
   - Save to `data/dataset_uniform.npz`.

2. **Phase 2b — Singularity Augmentation (50,000 samples)**:
   - Generate extra samples near known singularity configurations:
     - **Shoulder singularity**: θ2 ≈ 0 or ±π (arm fully extended/retracted)
     - **Elbow singularity**: θ3 ≈ 0 or ±π
     - **Wrist singularity**: θ5 ≈ 0 (wrist lock)
   - Use Gaussian perturbation around these singular configs (σ = 0.1 rad).
   - Save to `data/dataset_singularity.npz`.

3. **Phase 2c — Workspace Boundary Augmentation (25,000 samples)**:
   - Sample configurations near joint limits (within 10% of limit range from boundary).
   - Save to `data/dataset_boundary.npz`.

4. **Combine & Preprocess**:
   - Merge all datasets → `data/dataset_combined.npz`.
   - Compute normalization statistics (mean, std for inputs and outputs).
   - Save normalization params to `data/normalization_params.npz`.
   - Split: 70% train / 15% validation / 15% test.
   - Save splits to `data/train.npz`, `data/val.npz`, `data/test.npz`.

### File: `src/dataset.py`

PyTorch `Dataset` class `IKDataset`:
- Loads from `.npz` files.
- Applies normalization.
- Returns `(input_tensor, target_tensor)` pairs.
- `DataLoader` creation helper with configurable batch size.

### Verification:
- Print dataset statistics: min/max/mean/std per feature.
- Verify no NaN or Inf values.
- Verify FK round-trip on a sample of 100 data points (FK of outputs should ≈ inputs).
- Plot 3D scatter of generated end-effector positions to visualize workspace coverage.

**Update `context_log.json`** with dataset statistics.

---

## Phase 3: Neural Network Architecture & Training (ITERATIVE)

### File: `src/model.py`

Define multiple architectures to try iteratively:

#### Iteration 1: Baseline MLP
```python
class IKNet(nn.Module):
    # Input: 6 (pose) → Output: 6 (joint angles)
    # Layers: 6 → 256 → 256 → 256 → 256 → 6
    # Activation: ReLU
    # Dropout: 0.2
    # No batch norm
```

#### Iteration 2: Deeper + BatchNorm
```python
class IKNetV2(nn.Module):
    # 6 → 512 → 512 → 256 → 256 → 128 → 6
    # Activation: ReLU
    # BatchNorm after each hidden layer
    # Dropout: 0.15
```

#### Iteration 3: Residual Connections
```python
class IKNetV3(nn.Module):
    # Same as V2 but with skip/residual connections
    # Every 2 layers: x = F(x) + x (with projection if dims differ)
    # This helps gradient flow for deeper networks
```

#### Iteration 4: Sin/Cos Output Encoding
```python
class IKNetV4(nn.Module):
    # Output: 12 neurons (sin(θ_i), cos(θ_i) for i=1..6)
    # Then recover angles via atan2
    # This handles angle wrapping naturally
    # Same backbone as best previous iteration
```

#### Iteration 5: Multi-Head Architecture
```python
class IKNetV5(nn.Module):
    # Shared backbone → split into position-related joints (1-3) and orientation joints (4-6)
    # Each head has dedicated layers
    # Helps because joints 1-3 primarily control position, 4-6 control orientation
```

### File: `src/train.py`

Training loop with:

1. **Optimizer**: Adam, initial LR = 1e-3
2. **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
3. **Loss**: MSE (default), with option for custom weighted loss
4. **Early stopping**: patience=25 epochs monitoring val_loss
5. **Epochs**: max 500
6. **Batch size**: 256 (CPU-friendly)
7. **Checkpointing**: Save best model by val_loss to `models/`
8. **Logging**: Save training curves (train_loss, val_loss per epoch) to `results/`

**CPU Training Considerations**:
- Use `torch.set_num_threads(N)` where N = number of CPU cores
- Batch size 256 is optimal for CPU (not too big, not too small)
- Use `torch.compile()` if PyTorch 2.0+ for speedup
- Expected training time: ~15-40 min per iteration depending on architecture

**Iterative Training Strategy** (the core trial-and-error loop):

```
FOR each iteration (1 through 5+):
    1. Read context_log.json to see previous results
    2. Select/modify architecture based on previous metrics
    3. Train the model
    4. Evaluate on test set (Phase 4 metrics)
    5. Compare with previous iterations
    6. Log everything to context_log.json
    7. If position_RMSE < 1.0mm AND success_rate > 95%: STOP
    8. Otherwise: analyze failure modes and plan next iteration
       - If loss plateaus high → try deeper/wider network
       - If overfitting → increase dropout/regularization
       - If orientation errors high → try sin/cos encoding
       - If specific workspace regions fail → add more training data there
```

### Additional Training Tricks to Try:
- **Learning rate warmup** for first 5 epochs
- **Gradient clipping** at max_norm=1.0
- **Data augmentation**: Add small Gaussian noise (σ=1e-4) to inputs during training
- **Loss weighting**: Weight position components more if orientation is easier or vice-versa
- **Curriculum learning**: Train first on "easy" samples (away from singularities), then gradually include harder ones

### Verification per iteration:
- Training loss curve should decrease monotonically
- Validation loss should plateau or decrease (no divergence = no overfitting)
- After training, run full evaluation (Phase 4)

**Update `context_log.json`** after each iteration.

---

## Phase 4: Evaluation & Metrics

### File: `src/evaluate.py`

Implement comprehensive evaluation:

1. **Cartesian Position Error**:
   - Predict joint angles from test poses
   - Run FK on predicted joints → achieved pose
   - Compute: `E_pos = ||p_target - p_achieved||₂` (Euclidean distance in mm)
   - Report: mean, std, median, 95th percentile, max
   - Target: < 1.0 mm RMS

2. **Orientation Error**:
   - Compute angular difference between target and achieved orientations
   - Use geodesic distance on SO(3) for proper orientation error
   - Report: mean, std, median, 95th percentile, max (in degrees)
   - Target: < 0.5° RMS

3. **Joint Space RMSE**:
   - Direct comparison: predicted vs ground-truth joint angles
   - Report per-joint RMSE (degrees)

4. **Success Rate**:
   - % of test cases where position error < 1mm AND orientation error < 0.5°
   - Target: > 95%

5. **Inference Time**:
   - Time 1000 individual predictions, report mean/std/max (ms)
   - Also time batch prediction of 1000 samples at once
   - Target: < 1 ms per query

6. **Comparative Benchmark** (vs numerical IK):
   - Run `roboticstoolbox` numerical IK on same 1000 test poses
   - Measure its time and accuracy
   - Compute speedup factor: `time_numerical / time_nn`
   - Expected: 100-1000× speedup

7. **Workspace Coverage Analysis**:
   - Divide workspace into 3D grid (e.g., 10×10×10)
   - Compute mean position error per grid cell
   - Identify regions with high error
   - Generate workspace heatmap

8. **Singularity Performance**:
   - Evaluate specifically on singularity test samples
   - Compare NN vs numerical solver (which may fail near singularities)
   - Report: NN success rate vs numerical solver success rate in singular regions

9. **Trajectory Following**:
   - Generate circular, helical, and linear trajectories (100 points each)
   - Predict IK for each waypoint
   - Compute tracking error and joint smoothness (jerk metric)
   - Save trajectory data for visualization

### Output:
- Save all metrics to `results/metrics.json`
- Save detailed per-sample errors to `results/detailed_errors.csv`
- Generate plots and save to `results/` and `web/static/plots/`:
  - `training_loss_curves.png` — Train/val loss per epoch for all iterations
  - `position_error_histogram.png` — Distribution of position errors
  - `orientation_error_histogram.png` — Distribution of orientation errors
  - `workspace_heatmap.png` — 3D workspace color-coded by error
  - `per_joint_error.png` — Box plot of error per joint
  - `inference_time_comparison.png` — Bar chart: NN vs numerical solver
  - `trajectory_tracking.png` — Target vs achieved path for each trajectory
  - `iteration_comparison.png` — Metrics across all training iterations
  - `singularity_comparison.png` — NN vs numerical near singularities
  - `error_vs_distance.png` — Position error vs distance from workspace center
  - `joint_trajectory_plots.png` — Joint angle trajectories for demo paths

**Update `context_log.json`** with all metrics.

---

## Phase 5: Visualization & Arm Movement

### File: `src/visualization.py`

1. **3D Robot Arm Visualization**:
   - Given joint angles, compute all link positions using FK
   - Plot the 6-link arm in 3D (matplotlib)
   - Show: base, joints as spheres, links as lines/cylinders, end-effector as arrow
   - Save individual frames as PNG images

2. **Animated Arm Movement**:
   - For a trajectory, generate arm pose at each waypoint
   - Create animated GIF showing the arm moving along the trajectory
   - Save to `results/arm_animation_circular.gif`, `arm_animation_helical.gif`, etc.

3. **Comparison Animation**:
   - Side-by-side: target path (red dots) vs achieved path (blue dots)
   - Overlay on the arm animation

4. **Generate Static Plots for Web**:
   - All the plots from Phase 4
   - Additional: training progress plot, architecture comparison bar chart

All generated images/GIFs go to `web/static/plots/` for serving via the web dashboard.

---

## Phase 6: Web Dashboard

### File: `web/app.py` (Flask)

Flask server with these routes:

```python
GET /                  → Main dashboard page
GET /api/metrics       → JSON of all metrics from results/metrics.json
GET /api/predict       → Live IK prediction: ?x=..&y=..&z=..&roll=..&pitch=..&yaw=..
                         Returns: joint_angles, position_error, inference_time_ms
GET /api/trajectory    → Generate and return trajectory IK:
                         ?type=circle|helix|line&points=100
                         Returns: waypoints, joint_angles, errors per point
GET /api/arm-pose      → Return arm link positions for given joint angles
                         For 3D arm visualization in the browser
GET /api/iterations    → Return iteration comparison data
GET /api/random-demo   → Random pose prediction with full error analysis
```

### File: `web/templates/index.html` + `web/static/css/style.css` + `web/static/js/main.js`

**Design**: Modern, dark-themed dashboard with glassmorphism effects.

**Dashboard Sections** (all on one scrollable page):

#### 1. Header
- Project title: "Neural IK Solver — 6-DOF Manipulator"
- Subtitle with student names and course info
- Status badge: Best model accuracy

#### 2. Key Metrics Cards (Top Row)
- Position RMSE (mm) — large number + green/red indicator vs 1mm target
- Orientation RMSE (°) — large number + indicator vs 0.5° target
- Success Rate (%) — percentage with progress ring
- Avg Inference Time (ms) — with speedup factor vs numerical
- Training Iterations — count with best iteration highlighted

#### 3. Training Progress Section
- Interactive training loss curves (use Plotly.js for interactivity)
- Iteration comparison bar chart
- Architecture details table per iteration

#### 4. Error Analysis Section
- Position error histogram (Plotly)
- Orientation error histogram (Plotly)
- Per-joint error box plot
- Workspace heatmap image
- Error vs distance scatter plot

#### 5. Live Prediction Demo
- Input form: 6 fields (x, y, z, roll, pitch, yaw) with sliders
- "Predict" button → calls `/api/predict`
- Output: Predicted joint angles, position error, inference time
- **3D arm visualization** using Plotly 3D scatter/lines:
  - Shows the robot arm in the predicted configuration
  - Updates live when prediction is made
  - Rotatable/zoomable 3D view

#### 6. Trajectory Demo
- Dropdown: Circle / Helix / Linear path
- "Run Trajectory" button
- Shows animated arm movement (auto-playing frames with JS setInterval)
- Side panel: tracking error plot, joint angle plots

#### 7. Numerical vs NN Comparison
- Side-by-side table: accuracy, speed, success rate
- Speedup chart
- Singularity comparison section

#### 8. Arm Movement Gallery
- Embedded animated GIFs of arm following trajectories
- Before/after comparison images

#### 9. Iteration Log
- Expandable cards for each training iteration
- Shows: architecture, hyperparams, metrics, what changed, why

**Styling Requirements**:
- Dark background (#0a0a0f or similar deep dark)
- Glassmorphism cards (backdrop-filter: blur, semi-transparent backgrounds)
- Accent color: cyan/teal (#00d4ff) with purple (#8b5cf6) secondary
- Smooth gradient borders on cards
- Google Font: "Inter" for text, "JetBrains Mono" for numbers/code
- Hover animations on cards (subtle scale + glow)
- Smooth scroll between sections
- Mobile responsive (grid → stack on small screens)
- Loading skeletons while data fetches
- Micro-animations: number count-up for metrics, smooth chart transitions

---

## Phase 7: Master Script & End-to-End Execution

### File: `scripts/run_all.py`

Orchestrates the entire pipeline:

```python
"""
Master script — runs everything end-to-end.
Reads context_log.json to resume from last completed phase.
"""

def main():
    context = load_context_log()
    
    if not context['last_completed_phase'] or context['last_completed_phase'] < 'phase1':
        print("=== Phase 1: Robot Model Setup ===")
        # Verify FK engine works
        run_phase1_verification()
        update_context('phase1')
    
    if context['last_completed_phase'] < 'phase2':
        print("=== Phase 2: Data Generation ===")
        generate_all_datasets()
        update_context('phase2')
    
    if context['last_completed_phase'] < 'phase3':
        print("=== Phase 3: Iterative Training ===")
        for iteration in range(1, 6):
            if iteration <= len(context.get('iterations', [])):
                continue  # Already done
            model = create_model(iteration)
            train(model)
            metrics = evaluate(model)
            log_iteration(iteration, metrics)
            if meets_targets(metrics):
                print(f"✅ Targets met at iteration {iteration}!")
                break
        update_context('phase3')
    
    if context['last_completed_phase'] < 'phase4':
        print("=== Phase 4: Full Evaluation ===")
        run_full_evaluation()
        update_context('phase4')
    
    if context['last_completed_phase'] < 'phase5':
        print("=== Phase 5: Visualization ===")
        generate_all_visualizations()
        update_context('phase5')
    
    print("=== Phase 6: Web Dashboard Ready ===")
    print("Run: python web/app.py")
```

### File: `scripts/run_web.py`
```python
# Starts Flask dev server on port 5000
from web.app import app
app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## Execution Order Summary

| Step | Action | Expected Time (CPU) |
|------|--------|-------------------|
| 0 | Install dependencies | 2 min |
| 1 | Create robot model, verify FK | 1 min |
| 2 | Generate 175K training samples | 5-10 min |
| 3.1 | Train Iteration 1 (baseline MLP) | 15-25 min |
| 3.2 | Evaluate Iteration 1 | 2 min |
| 3.3 | Train Iteration 2 (deeper + BN) | 20-30 min |
| 3.4 | Evaluate Iteration 2 | 2 min |
| 3.5 | Train Iteration 3 (residual) | 20-30 min |
| 3.6 | Evaluate Iteration 3 | 2 min |
| 3.7 | Train Iteration 4 (sin/cos output) | 20-30 min |
| 3.8 | Evaluate Iteration 4 | 2 min |
| 3.9 | Train Iteration 5 (multi-head) | 20-30 min |
| 3.10 | Evaluate Iteration 5 | 2 min |
| 4 | Full evaluation + benchmarks | 5 min |
| 5 | Generate all visualizations + animations | 5 min |
| 6 | Build & launch web dashboard | 5 min |
| **Total** | | **~2-3 hours** |

---

## Key Reminders for Claude Code

1. **Always read `context_log.json` first** before doing anything. If it exists and has progress, resume from where it left off.

2. **Always update `context_log.json` after each phase** with full metrics and decisions.

3. **CPU training**: Use `device = torch.device('cpu')`. Set `torch.set_num_threads()` to available cores. Keep batch size at 256. Don't use any CUDA/GPU code.

4. **Error handling**: Wrap all major operations in try/except. If training fails, log the error to `context_log.json` and try the next architecture.

5. **File paths**: All relative to `/home/zypher/Music/Shi go down/`.

6. **Testing**: After each module, run a quick sanity check before moving on. Don't wait until the end to discover bugs.

7. **The web dashboard must work standalone** — once `results/metrics.json` and plots exist, `python web/app.py` should display everything even without retraining.

8. **Generate real data, real models, real results** — no mock data, no placeholder charts. Everything should be computed from actual training runs.

9. **If any iteration meets the targets** (position RMSE < 1mm, success rate > 95%), you can stop training early and proceed to evaluation. Don't force all 5 iterations if targets are already met.

10. **The animated 3D arm visualization in the web dashboard** should use Plotly.js 3D traces — plot line segments for links, spheres for joints, and update on prediction. This works without any backend rendering.

11. **Make the web dashboard beautiful** — this is a demo/presentation piece. Spend time on CSS, animations, and layout. Use the design specs above.

12. **All imports should use relative paths** within the `src/` package. The web app can import from `src/` by adding the project root to `sys.path`.

---

## Success Criteria

The project is complete when:

- [x] `context_log.json` shows at least 3 training iterations with improving metrics
- [x] Best model achieves position RMSE < 1.0 mm (or best achievable — document if target not met)
- [x] Best model achieves orientation RMSE < 0.5° (or best achievable)
- [x] Success rate > 95% (or best achievable)
- [x] Inference time < 1 ms per prediction
- [x] Speedup vs numerical solver documented (target: 100×+)
- [x] Web dashboard at `http://localhost:5000` shows:
  - All metrics with visual indicators
  - Training progress across iterations
  - Error distribution plots
  - Live prediction with 3D arm visualization
  - Trajectory demo with animation
  - Numerical vs NN comparison
  - Iteration history log
- [x] All generated plots saved in `web/static/plots/`
- [x] `context_log.json` fully populated with all iteration data
- [x] `python scripts/run_all.py` executes the full pipeline without errors
- [x] `python scripts/run_web.py` launches the dashboard successfully
