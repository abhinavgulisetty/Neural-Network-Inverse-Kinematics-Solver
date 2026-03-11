"""
Trajectory generation for circular, helical, and linear paths.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot_model import RobotModel


def generate_circular_trajectory(center=None, radius=0.1, n_points=100, plane='xy'):
    """
    Generate a circular trajectory in task space.

    Args:
        center: [x, y, z] center of circle (meters). If None, use robot workspace center.
        radius: radius of circle (meters)
        n_points: number of waypoints
        plane: 'xy', 'xz', or 'yz'

    Returns:
        waypoints: (n_points, 6) array of [x, y, z, roll, pitch, yaw]
    """
    if center is None:
        center = [0.4, 0.0, 0.4]

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    waypoints = np.zeros((n_points, 6))

    for i in range(n_points):
        if plane == 'xy':
            waypoints[i, 0] = center[0] + radius * np.cos(t[i])
            waypoints[i, 1] = center[1] + radius * np.sin(t[i])
            waypoints[i, 2] = center[2]
        elif plane == 'xz':
            waypoints[i, 0] = center[0] + radius * np.cos(t[i])
            waypoints[i, 1] = center[1]
            waypoints[i, 2] = center[2] + radius * np.sin(t[i])
        elif plane == 'yz':
            waypoints[i, 0] = center[0]
            waypoints[i, 1] = center[1] + radius * np.cos(t[i])
            waypoints[i, 2] = center[2] + radius * np.sin(t[i])

        waypoints[i, 3:] = [0.0, np.pi, 0.0]

    return waypoints


def generate_helical_trajectory(center=None, radius=0.1, height=0.2, n_points=100):
    """Generate a helical trajectory."""
    if center is None:
        center = [0.4, 0.0, 0.3]

    t = np.linspace(0, 4 * np.pi, n_points)
    waypoints = np.zeros((n_points, 6))

    for i in range(n_points):
        waypoints[i, 0] = center[0] + radius * np.cos(t[i])
        waypoints[i, 1] = center[1] + radius * np.sin(t[i])
        waypoints[i, 2] = center[2] + height * t[i] / (4 * np.pi)
        waypoints[i, 3:] = [0.0, np.pi, 0.0]

    return waypoints


def generate_linear_trajectory(start=None, end=None, n_points=100):
    """Generate a straight-line trajectory."""
    if start is None:
        start = np.array([0.3, -0.2, 0.3, 0.0, np.pi, 0.0])
    if end is None:
        end = np.array([0.5, 0.2, 0.5, 0.0, np.pi, 0.0])

    start = np.asarray(start)
    end = np.asarray(end)

    waypoints = np.zeros((n_points, 6))
    for i in range(n_points):
        alpha = i / (n_points - 1)
        waypoints[i] = start + alpha * (end - start)

    return waypoints


def get_trajectory(traj_type='circle', n_points=100):
    """Get a trajectory by type name."""
    if traj_type == 'circle':
        return generate_circular_trajectory(n_points=n_points)
    elif traj_type == 'helix':
        return generate_helical_trajectory(n_points=n_points)
    elif traj_type == 'line':
        return generate_linear_trajectory(n_points=n_points)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
