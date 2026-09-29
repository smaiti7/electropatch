"""Deterministic rigid-body DNA perturbations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    pose_id: str
    pose_type: str
    angle_degrees: float
    axis: np.ndarray
    translation: np.ndarray

    @property
    def translation_angstrom(self) -> float:
        return float(np.linalg.norm(self.translation))


def rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis must be nonzero")
    x, y, z = axis / norm
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    one_minus_c = 1 - c
    return np.array(
        [
            [c + x*x*one_minus_c, x*y*one_minus_c - z*s, x*z*one_minus_c + y*s],
            [y*x*one_minus_c + z*s, c + y*y*one_minus_c, y*z*one_minus_c - x*s],
            [z*x*one_minus_c - y*s, z*y*one_minus_c + x*s, c + z*z*one_minus_c],
        ]
    )


def transform(coords: np.ndarray, pose: Pose) -> np.ndarray:
    """Rotate the complete DNA partner around its centroid, then translate."""
    center = coords.mean(axis=0)
    return (coords - center) @ rotation_matrix(pose.axis, pose.angle_degrees).T + center + pose.translation


def generate_poses(n_decoys: int = 50, seed: int = 42) -> list[Pose]:
    if n_decoys <= 0 or n_decoys % 2:
        raise ValueError("n_decoys must be a positive even number")
    rng = np.random.default_rng(seed)
    poses = [Pose("native", "native", 0.0, np.array([1.0, 0.0, 0.0]), np.zeros(3))]
    for pose_type, count, angles, shifts in (
        ("near", n_decoys // 2, (2.0, 15.0), (0.5, 3.0)),
        ("hard", n_decoys // 2, (15.0, 40.0), (3.0, 8.0)),
    ):
        for number in range(1, count + 1):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            angle = float(rng.uniform(*angles))
            displacement = float(rng.uniform(*shifts))
            poses.append(Pose(f"{pose_type}_{number:02d}", pose_type, angle, axis, direction * displacement))
    return poses
