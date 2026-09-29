"""Run a fixed native-versus-decoy experiment for one protein-DNA complex."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import __version__
from .decoys import generate_poses, transform
from .scoring import calculate_features, score_variants
from .structures import load_complex


FIELDNAMES = [
    "pose_id", "pose_type", "rotation_degrees", "axis_x", "axis_y", "axis_z",
    "translation_x", "translation_y", "translation_z", "translation_angstrom",
    "contacts", "clashes", "basic_phosphate_contacts", "mean_nearest_distance",
    "geometry_score", "geometry_rank", "combined_score", "rank",
]


def _plot_ranking(rows: list[dict], path: Path, pdb_id: str) -> None:
    ranked = sorted(rows, key=lambda row: row["rank"])
    native = next(row for row in ranked if row["pose_id"] == "native")
    colors = ["#df6b2d" if row["pose_id"] == "native" else "#3d6482" for row in ranked]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(1, len(ranked) + 1), [float(row["combined_score"]) for row in ranked], color=colors, width=0.9)
    ax.set(xlabel="Pose rank (best to worst)", ylabel="Predefined combined score (z units)",
           title=f"{pdb_id} native rank: {native['rank']} of {len(ranked)}")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_features(rows: list[dict], path: Path, pdb_id: str) -> None:
    specs = [
        ("contacts", "Heavy-atom contacts ≤4.5 Å"),
        ("clashes", "Heavy-atom clashes ≤2.0 Å"),
        ("basic_phosphate_contacts", "Basic N–phosphate O contacts ≤4.5 Å"),
        ("mean_nearest_distance", "Mean nearest distance (Å)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8))
    for ax, (field, title) in zip(axes.flat, specs):
        groups = [[float(row[field]) for row in rows if row["pose_type"] == kind] for kind in ("near", "hard")]
        ax.boxplot(groups, positions=[1, 2], widths=0.4, showfliers=True)
        native = float(next(row[field] for row in rows if row["pose_type"] == "native"))
        ax.scatter([0], [native], color="#df6b2d", s=60, zorder=3)
        ax.set_xticks([0, 1, 2], ["Native\n(n=1)", f"Near\n(n={len(groups[0])})", f"Hard\n(n={len(groups[1])})"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{pdb_id} interface descriptors across native and rigid-body decoys", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    pdb_file: str | Path,
    output_dir: str | Path = ".",
    protein_chain: str | tuple[str, ...] = "A",
    dna_chains: tuple[str, ...] = ("B", "C"),
    n_decoys: int = 50,
    seed: int = 42,
    pdb_id: str | None = None,
    subdir: str | None = None,
) -> dict:
    pdb_file = Path(pdb_file)
    output_dir = Path(output_dir)
    pdb_id = pdb_id or pdb_file.stem.upper()
    protein_chains = (protein_chain,) if isinstance(protein_chain, str) else protein_chain
    protein, dna = load_complex(pdb_file, protein_chain, dna_chains)
    poses = generate_poses(n_decoys, seed)
    features = [
        calculate_features(protein, dna, dna.coords if pose.pose_type == "native" else transform(dna.coords, pose))
        for pose in poses
    ]
    geometry_scores, scores = score_variants(features)
    order = sorted(range(len(poses)), key=lambda i: (-scores[i], i))
    geometry_order = sorted(range(len(poses)), key=lambda i: (-geometry_scores[i], i))
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    geometry_ranks = {index: rank for rank, index in enumerate(geometry_order, 1)}
    rows = []
    for i, (pose, feature, score) in enumerate(zip(poses, features, scores)):
        rows.append({
            "pose_id": pose.pose_id,
            "pose_type": pose.pose_type,
            "rotation_degrees": f"{pose.angle_degrees:.8f}",
            "axis_x": f"{pose.axis[0]:.8f}",
            "axis_y": f"{pose.axis[1]:.8f}",
            "axis_z": f"{pose.axis[2]:.8f}",
            "translation_x": f"{pose.translation[0]:.8f}",
            "translation_y": f"{pose.translation[1]:.8f}",
            "translation_z": f"{pose.translation[2]:.8f}",
            "translation_angstrom": f"{pose.translation_angstrom:.8f}",
            "contacts": feature.contacts,
            "clashes": feature.clashes,
            "basic_phosphate_contacts": feature.basic_phosphate_contacts,
            "mean_nearest_distance": f"{feature.mean_nearest_distance:.8f}",
            "geometry_score": f"{geometry_scores[i]:.8f}",
            "geometry_rank": geometry_ranks[i],
            "combined_score": f"{score:.8f}",
            "rank": ranks[i],
        })
    results_dir = output_dir / "results" / subdir if subdir else output_dir / "results"
    figures_dir = output_dir / "figures" / subdir if subdir else output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "pose_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    _plot_ranking(rows, figures_dir / "pose_ranking.png", pdb_id)
    _plot_features(rows, figures_dir / "feature_comparison.png", pdb_id)
    native_rank = ranks[0]
    summary = {
        "project": "ElectroPatch",
        "version": __version__,
        "pdb_id": pdb_id,
        "pdb_sha256": hashlib.sha256(pdb_file.read_bytes()).hexdigest(),
        "protein_chains": list(protein_chains),
        "dna_chains": list(dna_chains),
        "chain_id_convention": "PDB author chain IDs",
        "protein_heavy_atoms": len(protein),
        "dna_heavy_atoms": len(dna),
        "seed": seed,
        "n_decoys": n_decoys,
        "n_poses": len(poses),
        "native_rank": native_rank,
        "native_geometry_rank": geometry_ranks[0],
        "native_percentile": round(100 * (len(poses) - native_rank) / (len(poses) - 1), 2),
        "decoys_below_native": len(poses) - native_rank,
        "score_formula": "z(contacts) + z(basic_phosphate_contacts) - 2*z(clashes) - z(mean_nearest_distance)",
        "zscore_definition": "population standard deviation over all poses; constant features contribute zero",
        "distance_definition": "mean nearest protein heavy-atom distance over all DNA heavy atoms",
        "contact_cutoff_angstrom": 4.5,
        "clash_cutoff_angstrom": 2.0,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
