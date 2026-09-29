"""Small, checksum-verified comparison across independent PDB complexes."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pipeline import run


COMPARISON_FIELDS = [
    "pdb_id", "system_name", "resolution_angstrom", "protein_chains", "dna_chains",
    "n_poses", "native_rank_geometry", "native_rank_combined", "rank_change_with_contact_descriptor",
    "native_percentile_combined", "native_contacts", "native_clashes", "native_basic_phosphate_contacts",
]


def _chains(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(","))


def run_comparison(
    manifest_file: str | Path = "data/manifest.csv",
    output_dir: str | Path = ".",
    n_decoys: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Run the same pose protocol per manifest row and summarize native ranks.

    Scores are z-standardized within each complex. The comparison uses ranks,
    never raw scores from different complexes.
    """
    manifest_file = Path(manifest_file)
    output_dir = Path(output_dir)
    with manifest_file.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    if not entries:
        raise ValueError("The comparison manifest is empty")
    if len({entry["pdb_id"] for entry in entries}) != len(entries):
        raise ValueError("The comparison manifest contains duplicate PDB IDs")
    rows = []
    for entry in entries:
        pdb_id = entry["pdb_id"].upper()
        pdb_file = manifest_file.parent / entry["filename"]
        digest = hashlib.sha256(pdb_file.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {pdb_id}: expected {entry['sha256']}, got {digest}")
        summary = run(
            pdb_file=pdb_file,
            output_dir=output_dir,
            protein_chain=_chains(entry["protein_chains"]),
            dna_chains=_chains(entry["dna_chains"]),
            n_decoys=n_decoys,
            seed=seed,
            pdb_id=pdb_id,
            subdir=pdb_id,
        )
        with (output_dir / "results" / pdb_id / "pose_scores.csv").open(newline="", encoding="utf-8") as handle:
            native = next(row for row in csv.DictReader(handle) if row["pose_id"] == "native")
        rows.append({
            "pdb_id": pdb_id,
            "system_name": entry["system_name"],
            "resolution_angstrom": entry["resolution_angstrom"],
            "protein_chains": entry["protein_chains"],
            "dna_chains": entry["dna_chains"],
            "n_poses": summary["n_poses"],
            "native_rank_geometry": summary["native_geometry_rank"],
            "native_rank_combined": summary["native_rank"],
            "rank_change_with_contact_descriptor": summary["native_geometry_rank"] - summary["native_rank"],
            "native_percentile_combined": summary["native_percentile"],
            "native_contacts": native["contacts"],
            "native_clashes": native["clashes"],
            "native_basic_phosphate_contacts": native["basic_phosphate_contacts"],
        })
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    with (results_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    overview = {
        "study_type": "exploratory pilot; selected protein-DNA complexes",
        "n_complexes": len(rows),
        "n_decoys_per_complex": n_decoys,
        "seed": seed,
        "scores_comparable_across_complexes": False,
        "comparison_basis": f"native ranks within each complex's fixed {n_decoys + 1}-pose candidate set",
        "median_native_rank_combined": median(row["native_rank_combined"] for row in rows),
        "native_top5_count_combined": sum(row["native_rank_combined"] <= 5 for row in rows),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    }
    (results_dir / "comparison.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    chunks = [rows[:4], rows[4:]] if len(rows) > 4 else [rows]
    fig, axes = plt.subplots(len(chunks), 1, figsize=(12.8, 3.9 * len(chunks)), squeeze=False)
    ceiling = max(max(row["native_rank_combined"], row["native_rank_geometry"]) for row in rows) + 5
    for ax, chunk in zip(axes.flat, chunks):
        positions = list(range(len(chunk)))
        ax.bar([x - 0.17 for x in positions], [row["native_rank_geometry"] for row in chunk], 0.34,
               color="#2563a6", label="Geometry")
        ax.bar([x + 0.17 for x in positions], [row["native_rank_combined"] for row in chunk], 0.34,
               color="#65b96f", label="Geometry + basic-phosphate contacts")
        ax.set_xticks(positions, [row["pdb_id"] + "\n" + fill(row["system_name"], width=24,
                        break_long_words=False, break_on_hyphens=False) for row in chunk])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0, ceiling)
        ax.grid(axis="y", alpha=0.2)
    fig.supylabel(f"Native rank among {n_decoys + 1} poses (lower is better)", fontsize=10)
    fig.suptitle(f"{len(rows)}-complex exploratory comparison", y=0.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.035, 0, 1, 0.89), h_pad=2.6)
    fig.savefig(figures_dir / "comparison.png", dpi=180)
    plt.close(fig)
    return rows
