"""Tests for rigid geometry, deterministic output, and PDB chain selection."""

import csv
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from electropatch.decoys import generate_poses, transform
from electropatch.comparison import run_comparison
from electropatch.pipeline import run
from electropatch.structures import load_complex


PDB = Path(__file__).resolve().parents[1] / "data" / "1AAY.pdb"
MANIFEST = PDB.parent / "manifest.csv"


class PipelineTests(unittest.TestCase):
    def test_transform_preserves_internal_distances(self):
        _, dna = load_complex(PDB)
        original = dna.coords[[0, 25, 100, 200, 400]]
        moved = transform(dna.coords, generate_poses(2, 42)[1])[[0, 25, 100, 200, 400]]
        self.assertTrue(np.allclose(np.linalg.norm(original[:, None] - original[None, :], axis=2),
                                    np.linalg.norm(moved[:, None] - moved[None, :], axis=2), atol=1e-10))

    def test_repeat_run_produces_identical_scores(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run(PDB, first, n_decoys=4, seed=42)
            b = run(PDB, second, n_decoys=4, seed=42)
            self.assertEqual(a, b)
            csv_a = (Path(first) / "results" / "pose_scores.csv").read_bytes()
            csv_b = (Path(second) / "results" / "pose_scores.csv").read_bytes()
            self.assertEqual(csv_a, csv_b)
            rows = list(csv.DictReader(csv_a.decode().splitlines()))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["pose_id"], "native")
            self.assertGreater(int(rows[0]["contacts"]), 0)

    def test_missing_chain_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Missing requested PDB chain"):
            load_complex(PDB, protein="Z")

    def test_multichain_partners_have_a_native_interface(self):
        from electropatch.scoring import calculate_features

        for code, protein_chains, dna_chains in (
            ("1LMB", ("3", "4"), ("1", "2")),
            ("1TRO", ("A", "C"), ("I", "J")),
            ("1A3Q", ("A", "B"), ("C", "D")),
            ("1HDD", ("C", "D"), ("A", "B")),
            ("1MNM", ("A", "B", "C", "D"), ("E", "F")),
            ("1RM1", ("A",), ("D", "E")),
            ("1PUE", ("E",), ("A", "B")),
        ):
            with self.subTest(pdb_id=code):
                protein, dna = load_complex(PDB.parent / f"{code}.pdb", protein_chains, dna_chains)
                features = calculate_features(protein, dna, dna.coords)
                self.assertGreater(features.contacts, 100)
                self.assertEqual(features.clashes, 0)

    def test_comparison_reproduces_and_checks_input_hashes(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_comparison(MANIFEST, first, n_decoys=4, seed=42)
            b = run_comparison(MANIFEST, second, n_decoys=4, seed=42)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 8)
            self.assertEqual((Path(first) / "results" / "comparison.csv").read_bytes(),
                             (Path(second) / "results" / "comparison.csv").read_bytes())
            bad_manifest = Path(first) / "bad_manifest.csv"
            with MANIFEST.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            row["filename"] = str(PDB)
            row["sha256"] = "0" * 64
            with bad_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run_comparison(bad_manifest, first, n_decoys=4, seed=42)

    def test_colab_notebook_is_valid_and_code_compiles(self):
        notebook = Path(__file__).resolve().parents[1] / "colab" / "ElectroPatch_Comparison.ipynb"
        data = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(data["nbformat"], 4)
        code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 5)
        for index, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Install the optional ML extra to test grouped ranking")
    def test_grouped_ml_is_deterministic_and_excludes_held_out_complex(self):
        from electropatch.ml import run_ml

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for folder in (first, second):
                run_comparison(MANIFEST, folder, n_decoys=4, seed=42)
            a = run_ml(MANIFEST, first)
            b = run_ml(MANIFEST, second)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 8)
            self.assertTrue(all(row["training_complexes"] == 7 for row in a))
            self.assertEqual((Path(first) / "results" / "ml_pose_scores.csv").read_bytes(),
                             (Path(second) / "results" / "ml_pose_scores.csv").read_bytes())
            with (Path(first) / "results" / "ml_pose_scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 40)
            self.assertEqual(sum(row["native_label"] == "1" for row in rows), 8)


if __name__ == "__main__":
    unittest.main()
