"""Exploratory native-pose ranking with complex-grouped cross-validation."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURE_NAMES = (
    "contacts_per_dna_atom", "clashes_per_dna_atom",
    "basic_phosphate_contacts_per_dna_atom", "mean_nearest_distance",
)


def run_ml(manifest_file: str | Path = "data/manifest.csv", output_dir: str | Path = ".") -> list[dict]:
    """Fit a fixed logistic model in leave-one-complex-out folds.

    Requires prior `electropatch compare` outputs. All poses from a held-out
    complex are excluded from fitting the StandardScaler and classifier.
    """
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install the optional ML dependencies with `pip install -e '.[ml]'`") from exc

    manifest_file, output_dir = Path(manifest_file), Path(output_dir)
    with manifest_file.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    if len(entries) < 3 or len({row["pdb_id"] for row in entries}) != len(entries):
        raise ValueError("ML requires at least three distinct complexes in the manifest")

    comparison_file = output_dir / "results" / "comparison.csv"
    overview_file = output_dir / "results" / "comparison.json"
    with overview_file.open(encoding="utf-8") as handle:
        overview = json.load(handle)
    if overview["manifest_sha256"] != hashlib.sha256(manifest_file.read_bytes()).hexdigest():
        raise ValueError("Comparison results were generated with a different manifest; rerun `electropatch compare`")
    with comparison_file.open(newline="", encoding="utf-8") as handle:
        baseline = {row["pdb_id"]: row for row in csv.DictReader(handle)}
    if set(baseline) != {row["pdb_id"] for row in entries}:
        raise ValueError("Comparison CSV and manifest contain different complexes")

    samples, groups, labels, identities = [], [], [], []
    for entry in entries:
        code = entry["pdb_id"]
        summary_file = output_dir / "results" / code / "summary.json"
        with summary_file.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary["pdb_sha256"] != entry["sha256"] or summary["seed"] != overview["seed"] or summary["n_decoys"] != overview["n_decoys_per_complex"]:
            raise ValueError(f"Stale or mismatched results for {code}; rerun `electropatch compare`")
        with (output_dir / "results" / code / "pose_scores.csv").open(newline="", encoding="utf-8") as handle:
            poses = list(csv.DictReader(handle))
        if len(poses) != summary["n_poses"] or sum(p["pose_id"] == "native" for p in poses) != 1:
            raise ValueError(f"Expected exactly one native among {summary['n_poses']} poses for {code}")
        dna_atoms = summary["dna_heavy_atoms"]
        for pose in poses:
            samples.append([
                float(pose["contacts"]) / dna_atoms,
                float(pose["clashes"]) / dna_atoms,
                float(pose["basic_phosphate_contacts"]) / dna_atoms,
                float(pose["mean_nearest_distance"]),
            ])
            groups.append(code)
            labels.append(int(pose["pose_id"] == "native"))
            identities.append((code, pose["pose_id"], pose["pose_type"]))

    x, y, group_array = np.asarray(samples), np.asarray(labels), np.asarray(groups)
    probabilities = np.full(len(y), np.nan)
    fold_training_counts = {}
    for training, held_out in LeaveOneGroupOut().split(x, y, groups=group_array):
        code = group_array[held_out[0]]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs"),
        )
        model.fit(x[training], y[training])
        probabilities[held_out] = model.predict_proba(x[held_out])[:, 1]
        fold_training_counts[code] = len(set(group_array[training]))
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The ML model produced non-finite probabilities")

    results_dir, figures_dir = output_dir / "results", output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pose_rows, comparison_rows = [], []
    for entry in entries:
        code = entry["pdb_id"]
        indices = np.flatnonzero(group_array == code)
        order = indices[np.argsort(-probabilities[indices], kind="stable")]
        ranks = {int(index): rank for rank, index in enumerate(order, start=1)}
        native_index = next(index for index in indices if labels[index] == 1)
        for index in indices:
            _, pose_id, pose_type = identities[index]
            pose_rows.append(dict(pdb_id=code, pose_id=pose_id, pose_type=pose_type,
                                  native_label=labels[index], held_out_native_probability=f"{probabilities[index]:.8f}",
                                  held_out_rank=ranks[int(index)]))
        comparison_rows.append(dict(pdb_id=code, system_name=entry["system_name"], n_poses=len(indices),
                                    training_complexes=fold_training_counts[code],
                                    native_rank_geometry=baseline[code]["native_rank_geometry"],
                                    native_rank_combined=baseline[code]["native_rank_combined"],
                                    native_rank_ml=ranks[int(native_index)],
                                    native_probability=f"{probabilities[native_index]:.8f}"))

    with (results_dir / "ml_pose_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pose_rows[0]))
        writer.writeheader()
        writer.writerows(pose_rows)
    with (results_dir / "ml_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    details = dict(
        study_type="exploratory synthetic-decoy ranking; not external predictive validation",
        protocol="LeaveOneGroupOut by PDB complex; all 51 poses of the held-out complex excluded from fitting",
        training_features=list(FEATURE_NAMES),
        model="StandardScaler then LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000)",
        feature_scaling="contact counts divided by the number of DNA heavy atoms; StandardScaler fitted on training folds only",
        label_definition="experimental pose=1; generated rigid-body decoys=0",
        n_complexes=len(entries), n_samples=len(y), n_native=int(y.sum()),
        median_native_rank_ml=median(row["native_rank_ml"] for row in comparison_rows),
        native_top5_count_ml=sum(row["native_rank_ml"] <= 5 for row in comparison_rows),
        manifest_sha256=overview["manifest_sha256"],
        comparison_csv_sha256=hashlib.sha256(comparison_file.read_bytes()).hexdigest(),
        scikit_learn_version=sklearn.__version__,
    )
    (results_dir / "ml_summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    chunks = [comparison_rows[:4], comparison_rows[4:]] if len(comparison_rows) > 4 else [comparison_rows]
    fig, axes = plt.subplots(len(chunks), 1, figsize=(12.8, 3.9 * len(chunks)), squeeze=False)
    ceiling = max(max(int(r["native_rank_combined"]), r["native_rank_ml"]) for r in comparison_rows) + 5
    for ax, chunk in zip(axes.flat, chunks):
        positions = np.arange(len(chunk))
        ax.bar(positions - 0.17, [int(r["native_rank_combined"]) for r in chunk], 0.34,
               color="#2563a6", label="Fixed combined score")
        ax.bar(positions + 0.17, [r["native_rank_ml"] for r in chunk], 0.34,
               color="#65b96f", label="Held-out ML")
        ax.set_xticks(positions, [r["pdb_id"] + "\n" + fill(r["system_name"], width=24,
                        break_long_words=False, break_on_hyphens=False) for r in chunk])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0, ceiling)
        ax.grid(axis="y", alpha=0.2)
    fig.supylabel(f"Native rank among {len(indices)} poses (lower is better)", fontsize=10)
    fig.suptitle("Leave-one-complex-out pilot comparison", y=0.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.035, 0, 1, 0.89), h_pad=2.6)
    fig.savefig(figures_dir / "ml_comparison.png", dpi=180)
    plt.close(fig)
    return comparison_rows
