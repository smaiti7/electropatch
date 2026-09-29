# ElectroPatch: reproducibility record

This record describes the committed eight-complex comparison and leave-one-complex-out ML pilot. Run every command from the repository root. The [analysis](ANALYSIS.md) interprets the measured results; [GitHub upload instructions](../GITHUB_UPLOAD.md) cover publication and personal details.

## 1. Frozen inputs

Eight coordinate files were downloaded from the RCSB PDB archive on **29 September 2026**. `data/manifest.csv` contains each source URL, X-ray resolution, selected PDB **author** chain IDs, selection note and SHA-256. `electropatch compare` verifies every file against this manifest before calculation. Inspect the actual manifest for the authoritative values.

| File | Protein chains | DNA chains | SHA-256 |
| --- | --- | --- | --- |
| `data/1AAY.pdb` | A | B,C | `303d41b3940088d157aceae7a80975e7ae45fd5e9b4b31484a32e217d06e1a04` |
| `data/1LMB.pdb` | 3,4 | 1,2 | `a61e5ae05fe2c588dc4c86943774f90b6610313cea8040e1b0189dc5c0548257` |
| `data/1TRO.pdb` | A,C | I,J | `636197b45e02cd1d2356b6cb376cacb0ff6086626543795e81ae131e50d948b0` |
| `data/1A3Q.pdb` | A,B | C,D | `7edb9f0989d29426b8de3720682c08efe25cdf9cbff130e4e10cba9248c51da0` |
| `data/1HDD.pdb` | C,D | A,B | `ed67971db072a4465dfab2e020ba1c3ce509057fc58a059e0ff1dd8eed179b42` |
| `data/1MNM.pdb` | A,B,C,D | E,F | `6128d6eea5e5f1c6c6ea78627e028a5347842c3383b5c47635a480dffa6674df` |
| `data/1RM1.pdb` | A | D,E | `ad257d01fc769339de78ebffe0d178efbf87eda158877997bb7c4123085b1d77` |
| `data/1PUE.pdb` | E | A,B | `f2431a3ccae8b0e30a6a21fa796f15b092e795ea1613dd31d76425e9f66aa8ef` |

Verify the files with `sha256sum data/*.pdb data/manifest.csv`. The parser reads `ATOM` records from the first model, removes hydrogens/deuterium, resolves alternate locations by occupancy, and excludes `HETATM`. This means zinc ions in 1AAY and possible other cofactors are not modeled. Selected heavy-atom counts and source checksums appear in `results/<PDB_ID>/summary.json`. Two-copy structures 1TRO and 1PUE use one selected copy each; 1RM1 scores the DNA-contacting TBP chain A.

## 2. Software and setup

The committed numeric outputs were generated with Python **3.12.3**, NumPy **1.26.4**, SciPy **1.11.4**, Matplotlib **3.6.3**, and scikit-learn **1.9.1**. Versions for the fixed-score run are recorded per structure; `results/ml_summary.json` records scikit-learn's version. The notebook installs pandas for display only. Dependencies in `pyproject.toml` specify supported ranges, so future environments can produce slightly different decimals or tied ranks.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml]'
python -m unittest discover -s tests -v
```

The test suite checks deterministic rigid transforms, chain errors, native interfaces, manifest-hash rejection, notebook syntax, grouped ML determinism and training group counts. The GitHub Actions workflow repeats installation, tests, the single-complex command and both comparative commands on Python 3.12.

## 3. Run the fixed comparison

Use the frozen defaults of **50 decoys per complex, seed 42**:

```bash
electropatch compare --manifest data/manifest.csv --n-decoys 50 --seed 42 --output-dir .
```

Expected geometry/combined ranks for PDBs in manifest order are `1AAY 18/19`, `1LMB 9/8`, `1TRO 6/19`, `1A3Q 22/26`, `1HDD 15/1`, `1MNM 3/4`, `1RM1 12/11`, and `1PUE 8/11` out of 51. This writes `results/comparison.csv`, `results/comparison.json`, `figures/comparison.png`, plus `pose_scores.csv`, `summary.json`, `pose_ranking.png` and `feature_comparison.png` for each complex.

For a clean rerun without touching committed artifacts:

```bash
repro_dir=$(mktemp -d /tmp/electropatch-repro.XXXXXX)
electropatch compare --manifest data/manifest.csv --n-decoys 50 --seed 42 --output-dir "$repro_dir"
```

The fixed-score implementation applies SciPy `cKDTree` to heavy-atom pairs. There are 25 near rigid-body transforms (2–15° and 0.5–3 Å) and 25 harder transforms (15–40° and 3–8 Å). Rotations use the selected DNA centroid. No candidate is rejected or relaxed. The feature definitions and score equations are specified in [ANALYSIS.md](ANALYSIS.md); every transform, descriptor and score is in the per-pose CSV. Each of the 51-pose sets is z-standardized separately with population standard deviation (`ddof=0`), and a constant feature contributes zero. Descending scores define ranks, with generation order resolving ties.

## 4. Run the ML analysis

After `compare`, run:

```bash
electropatch ml --manifest data/manifest.csv --output-dir .
```

For an isolated rerun, use `--output-dir "$repro_dir"` after the isolated `compare` command. The ML command checks that `comparison.json` matches the current manifest and that per-complex summary hashes, seeds and decoy counts match. It requires exactly one native and the expected pose count per group. Outputs are `results/ml_pose_scores.csv`, `results/ml_comparison.csv`, `results/ml_summary.json` and `figures/ml_comparison.png`.

The method uses four descriptors: contacts/DNA-atom count, clashes/DNA-atom count, basic-phosphate contacts/DNA-atom count and mean nearest distance in Å. These fixed transformations use no labels. For each `LeaveOneGroupOut` fold, a `StandardScaler` and class-balanced logistic regression (`C=1`, `lbfgs`, `max_iter=1000`) are fitted on **seven PDB complexes** and predict the eighth. There is no model selection or feature selection on held-out outcomes. The model's probabilities are used to rank poses within the held-out complex; they are not calibrated probabilities. See [ANALYSIS.md](ANALYSIS.md) for the central synthetic-decoy limitation.

Expected ML native ranks in manifest order are `4, 1, 4, 1, 1, 1, 2, 5` out of 51. The eight folds contain 408 pose samples overall, eight positive natives and 400 synthetic negatives. Each fold trains on 7 groups (357 samples).

## 5. Verify numeric artifacts

With the same numerical environment, these comparisons should be byte-for-byte identical:

```bash
cmp results/comparison.csv "$repro_dir/results/comparison.csv"
cmp results/comparison.json "$repro_dir/results/comparison.json"
cmp results/ml_comparison.csv "$repro_dir/results/ml_comparison.csv"
cmp results/ml_pose_scores.csv "$repro_dir/results/ml_pose_scores.csv"
cmp results/ml_summary.json "$repro_dir/results/ml_summary.json"
for pdb_id in 1AAY 1LMB 1TRO 1A3Q 1HDD 1MNM 1RM1 1PUE; do
  cmp "results/$pdb_id/pose_scores.csv" "$repro_dir/results/$pdb_id/pose_scores.csv"
  cmp "results/$pdb_id/summary.json" "$repro_dir/results/$pdb_id/summary.json"
done
```

Release artifact checksums (SHA-256):

| Artifact | SHA-256 |
| --- | --- |
| `data/manifest.csv` | `0844f14b6f6ff0950c9ca9041c80e2fc090a68e2d6215208b80ea3212f768624` |
| `results/comparison.csv` | `b9d89c7ef380e2ffdfff7433bfce7eb28fd2146ffd7b2d24cb3f7c2a0b8334ad` |
| `results/comparison.json` | `4091bae0b5e568d922b64b342bf008e5f733be0e4b7b902aa1c5d33b91bc6bb5` |
| `results/ml_comparison.csv` | `154d41c0bae79bbafbb098cd8a768d6d5be850c85cc10cfaafdf4726fb7d3f49` |
| `results/ml_pose_scores.csv` | `e277dec925e8a715620ac314a92e1b088750cf0f34e0fe913f6bf639862265db` |
| `results/ml_summary.json` | `f22ed2b4a59ea75f7d99281e98de18f70dd0d4720308cc4b2709ad35867c4a2e` |
| `results/1AAY/pose_scores.csv` | `303bd1f6a44f8dd26bd5a112375cfc6ec81c50e64a1730159b5f247d6f1ff389` |
| `results/1LMB/pose_scores.csv` | `9379a1a1fa150df35cd1a59231a556d24dd064228de04e4e9c469f37f49f6247` |
| `results/1TRO/pose_scores.csv` | `e9e2d77dd63b3c40a448ebe34ea9ecce17fa87bef062079a50253bfbede75e18` |
| `results/1A3Q/pose_scores.csv` | `7469d8ad2038e73a6a7cd0de4c6741ac7f9b84ec18933ef9b0fae4ddfc9625a5` |
| `results/1HDD/pose_scores.csv` | `4b1f97947fd700325dcb68377823403702cf6d677f502ad095606f752480182e` |
| `results/1MNM/pose_scores.csv` | `1b61fb722e2c5278dfb58bf3ab68d3e84f1ad74160269b11ade77000f71d8eae` |
| `results/1RM1/pose_scores.csv` | `437601f44a96c17cbdcd11a1f6bfa13cc70c1e71337e56d7377a2105e764f787` |
| `results/1PUE/pose_scores.csv` | `1b258cb09f95d8504852548a7a913eb489cd6827174630b6d80efe94192dc906` |

PNG bytes can vary with Matplotlib and rendering versions while showing the same data. In the committed figures, blue is the first score and light green the second; **four complexes appear on each of two rows**, with PDB IDs and full protein names. If numeric byte comparisons differ, check the bundled input hashes, selected chains, seed, package versions, per-pose scores and any tied ranks before interpreting a discrepancy.

## 6. Colab route

After the repository is public, replace `YOUR_USERNAME` in the README badge and open the notebook. Choose **Runtime → Run all**, paste the public GitHub URL when prompted, and inspect the tables and inline figures. The notebook clones the repository, prints the Git commit, installs this same package with the `[ml]` extra, and runs both comparison commands with the frozen manifest. The optional final cell packages `results/` and `figures/` as a ZIP. The notebook was syntax-checked and its Python calculation cells were run locally; a hosted Colab run requires the public GitHub repository and should be checked after push.

Record the pushed code revision with `git rev-parse HEAD`. Colab's printed commit should match it. Colab session storage is temporary; download the ZIP to retain generated outputs.

## 7. Development and execution history

1. Ran the predefined fixed score on 1AAY, then added 1LMB and 1TRO with explicit PDB author-chain selections and a descriptive geometry ablation.
2. Added five more X-ray structures from different DNA-binding systems: 1A3Q, 1HDD, 1MNM, 1RM1 and 1PUE. Checked each chain type and 4.5 Å native interface, selected one crystallographic copy where appropriate, froze coordinates and hashes, then ran the unchanged decoy/scoring protocol across all eight.
3. Added a fixed logistic model with complex-grouped cross-validation. The model pipeline fits preprocessing only on training groups. Generated per-pose predictions, held-out rank table and a separate figure; recorded both the apparent improvement and its strong synthetic-decoy limitations.
4. Expanded the Colab notebook to run and visualize the eight-complex comparison and optional grouped ML. Updated the comparison figures to two rows of four while preserving full names and blue/light-green colors.
5. Ran the test suite and locally executed both comparative commands. Reproduced the committed numerical outputs in an isolated output directory, inspected the exported figures, and wrote this record separately from the scientific analysis.

The repository is ready for author review and GitHub upload. The data and results are exploratory and should be presented as such in an application or interview.
