import json
import os
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONTEXT_LOG_PATH = PROJECT_ROOT / "context_log.json"


def load_context_log():
    if CONTEXT_LOG_PATH.exists():
        with open(CONTEXT_LOG_PATH, 'r') as f:
            return json.load(f)
    return {
        "last_completed_phase": None,
        "timestamp": None,
        "iterations": [],
        "best_model": {"iteration": 0, "path": None, "metrics": {}},
        "decisions": [],
        "dataset_stats": {},
        "phase_log": []
    }


def save_context_log(context):
    context["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(CONTEXT_LOG_PATH, 'w') as f:
        json.dump(context, f, indent=2, default=str)


def update_phase(phase_id, description=""):
    ctx = load_context_log()
    ctx["last_completed_phase"] = phase_id
    ctx["phase_log"].append({
        "phase": phase_id,
        "description": description,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    })
    save_context_log(ctx)


def log_iteration(iteration_num, architecture, hyperparams, metrics, changes, next_steps):
    ctx = load_context_log()
    entry = {
        "iteration": iteration_num,
        "architecture": architecture,
        "hyperparameters": hyperparams,
        "metrics": metrics,
        "changes_made": changes,
        "next_steps": next_steps
    }
    existing = [i for i, it in enumerate(ctx["iterations"]) if it["iteration"] == iteration_num]
    if existing:
        ctx["iterations"][existing[0]] = entry
    else:
        ctx["iterations"].append(entry)

    if metrics.get("position_rmse_mm") is not None:
        best_pos = ctx["best_model"].get("metrics", {}).get("position_rmse_mm", float('inf'))
        if metrics["position_rmse_mm"] < best_pos:
            ctx["best_model"] = {
                "iteration": iteration_num,
                "path": f"models/best_model_iter{iteration_num}.pth",
                "metrics": metrics
            }
    save_context_log(ctx)


def log_decision(decision):
    ctx = load_context_log()
    ctx["decisions"].append(decision)
    save_context_log(ctx)


def log_dataset_stats(stats):
    ctx = load_context_log()
    ctx["dataset_stats"] = stats
    save_context_log(ctx)


class Normalizer:

    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def fit(self, inputs, outputs):
        self.input_mean = np.mean(inputs, axis=0)
        self.input_std = np.std(inputs, axis=0)
        self.input_std[self.input_std < 1e-8] = 1.0

        self.output_mean = np.mean(outputs, axis=0)
        self.output_std = np.std(outputs, axis=0)
        self.output_std[self.output_std < 1e-8] = 1.0

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def normalize_output(self, y):
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y_norm):
        return y_norm * self.output_std + self.output_mean

    def denormalize_input(self, x_norm):
        return x_norm * self.input_std + self.input_mean

    def save(self, path):
        np.savez(path,
                 input_mean=self.input_mean, input_std=self.input_std,
                 output_mean=self.output_mean, output_std=self.output_std)

    def load(self, path):
        data = np.load(path)
        self.input_mean = data['input_mean']
        self.input_std = data['input_std']
        self.output_mean = data['output_mean']
        self.output_std = data['output_std']


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
