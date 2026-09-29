# ElectroPatch: eight-complex scientific analysis

## Question, scope and provenance

Can simple interface descriptors rank an experimental protein–DNA pose above 50 reproducible rigid-body perturbations? Does a basic-residue/phosphate proximity term help a fixed geometric score? Can a deliberately small, grouped machine-learning model rank held-out complexes better on the **same synthetic-decoy task**?

This is an exploratory pilot. The fixed combined-score formula was specified for the initial 1AAY experiment. The geometry-only ablation and additional PDB systems were added after that first run. The grouped ML experiment was then added after observing the heuristic comparison. Consequently, neither the expanded set nor the ML experiment is a prospective external validation. No fixed-score weights or ML hyperparameters were tuned against the reported held-out ranks.

All eight source entries are X-ray structures from [RCSB PDB](https://www.rcsb.org/). The exact downloaded files and author-chain selections are in `data/manifest.csv`; the command checks SHA-256 before use.

| PDB | Protein–DNA system | Resolution | Protein chains | DNA chains | Selected heavy atoms, protein / DNA |
| --- | --- | ---: | --- | --- | ---: |
| [1AAY](https://www.rcsb.org/structure/1AAY) | Zif268 zinc finger | 1.60 Å | A | B,C | 712 / 445 |
| [1LMB](https://www.rcsb.org/structure/1LMB) | Lambda repressor | 1.80 Å | 3,4 | 1,2 | 1,380 / 814 |
| [1TRO](https://www.rcsb.org/structure/1TRO) | Trp repressor | 1.90 Å | A,C | I,J | 1,632 / 772 |
| [1A3Q](https://www.rcsb.org/structure/1A3Q) | NF-kappa-B p52 | 2.10 Å | A,B | C,D | 4,518 / 445 |
| [1HDD](https://www.rcsb.org/structure/1HDD) | Engrailed homeodomain | 2.80 Å | C,D | A,B | 962 / 855 |
| [1MNM](https://www.rcsb.org/structure/1MNM) | MATalpha2/MCM1 complex | 2.25 Å | A,B,C,D | E,F | 2,544 / 1,060 |
| [1RM1](https://www.rcsb.org/structure/1RM1) | TBP/TFIIA/TATA-box DNA | 2.50 Å | A | D,E | 1,416 / 735 |
| [1PUE](https://www.rcsb.org/structure/1PUE) | PU.1 ETS domain | 2.10 Å | E | A,B | 738 / 650 |

A native-contact check at 4.5 Å found at least 312 cross-partner heavy-atom pairs for every selected system, with zero ≤2 Å clashes in each native. The 1TRO file contains another protein/DNA crystal copy; A,C/I,J selects one set. The 1PUE file also has two copies; E/A,B selects one. For 1RM1, chain A is the DNA-contacting TBP partner; TFIIA B and C have zero and only 22 4.5 Å cross-contacts with D/E respectively, so they are omitted from the scored protein partner. 1MNM retains all four contacting protein chains. These coordinate checks do not prove biological-assembly completeness; that requires manual structural curation.

## Fixed decoys and scoring

For each complex, protein coordinates stay fixed and the selected DNA duplex is rigidly transformed about its centroid. NumPy `default_rng(42)` creates 25 near decoys (rotation 2–15°, translation 0.5–3 Å) and 25 harder decoys (15–40°, 3–8 Å). No decoy is rejected, relaxed or energy minimized. Every transformation is stored in the per-pose CSV.

Four features are calculated from selected `ATOM` heavy atoms: cross-partner contacts at ≤4.5 Å; clashes at ≤2.0 Å; Lys NZ or Arg NE/NH1/NH2 to DNA phosphate OP1/OP2 (or O1P/O2P) contacts at ≤4.5 Å; and the mean nearest protein heavy-atom distance for **all** DNA heavy atoms. Within each 51-pose set, population z-scores are calculated separately. A constant feature contributes zero.

`S_combined = z(contacts) − 2 z(clashes) + z(basic-phosphate contacts) − z(mean nearest distance)`

`S_geometry = z(contacts) − 2 z(clashes) − z(mean nearest distance)`

Higher score ranks first; ties keep pose-generation order. These are heuristic scores, not binding energies. The basic-phosphate count is a geometric proxy for charge proximity. Raw scores and z-scores are not comparable across systems because each complex has its own candidate pool.

## Fixed-score results

| PDB | Geometry native rank | Combined native rank | Basic-phosphate effect | Native contacts | Native basic-phosphate pairs |
| --- | ---: | ---: | --- | ---: | ---: |
| 1AAY | 18/51 | 19/51 | 1 worse | 394 | 17 |
| 1LMB | 9/51 | 8/51 | 1 better | 432 | 8 |
| 1TRO | 6/51 | 19/51 | 13 worse | 412 | 13 |
| 1A3Q | 22/51 | 26/51 | 4 worse | 420 | 9 |
| 1HDD | 15/51 | 1/51 | 14 better | 312 | 25 |
| 1MNM | 3/51 | 4/51 | 1 worse | 1,135 | 39 |
| 1RM1 | 12/51 | 11/51 | 1 better | 487 | 14 |
| 1PUE | 8/51 | 11/51 | 3 worse | 336 | 16 |

The combined score has median rank 11/51 and places two of eight natives in the top five. The basic-phosphate term helps three systems and worsens five. The 1HDD improvement is large, while 1TRO and 1A3Q degrade. These eight selected structures do not justify an inference that the contact proxy is generally useful or harmful.

### Concrete failure modes

The top combined-score decoys for 1TRO and 1A3Q have 431 and 327 clashes respectively, yet they outrank their experimental poses. Their contact counts (4,602 and 3,340) compensate for the fixed clash penalty. In 1TRO, `hard_05` has a geometry score of −0.733 but a combined score of 2.792 because the standardized basic-phosphate feature adds +3.525. This is a failure of simple pair counting and weight balance, not evidence against physical electrostatics. Other top decoys include 1AAY `hard_23` (78 clashes), 1LMB `near_22` (50), 1MNM `near_19` (72), 1RM1 `near_16` (13) and 1PUE `near_15` (17). The 1HDD native is the top combined pose; its best competing decoy `near_11` has no ≤2 Å clashes. Detailed pose-level values are in `results/<PDB_ID>/pose_scores.csv`.

## Grouped ML pilot

The ML analysis reads exactly the comparison outputs above. It divides contacts, clashes and basic-phosphate counts by the selected DNA heavy-atom count to reduce scale differences, keeps mean nearest distance in Å, then fits a `StandardScaler` and class-balanced `LogisticRegression(C=1, solver='lbfgs', max_iter=1000)`. One native is labeled 1 and 50 generated decoys are labeled 0 per complex. `LeaveOneGroupOut` uses the **PDB ID** as the group: all 51 poses of one complex are held out, and both scaler and logistic model are fitted only on the other seven complexes. Predictions for the held-out 51 poses are ranked within that complex. No pose from the held-out complex enters training, feature scaling or hyperparameter selection. Division by DNA atom count is a fixed, label-free transformation.

| PDB held out | Training complexes | Fixed combined rank | ML rank |
| --- | ---: | ---: | ---: |
| 1AAY | 7 | 19 | 4 |
| 1LMB | 7 | 8 | 1 |
| 1TRO | 7 | 19 | 4 |
| 1A3Q | 7 | 26 | 1 |
| 1HDD | 7 | 1 | 1 |
| 1MNM | 7 | 4 | 1 |
| 1RM1 | 7 | 11 | 2 |
| 1PUE | 7 | 11 | 5 |

Median ML rank is 1.5/51; all eight natives enter the top five. This describes **only these eight held-out groups against this one decoy generator**. The 8 positives are too few for reliable performance estimates or uncertainty intervals, and the 400 negative examples are correlated poses generated by one procedure. Grouping prevents the most direct pose-level leakage, but shared generator artifacts remain. All eight experimental poses have zero ≤2 Å clashes, while many decoys are severely clashing; this makes the classification task artificially easy. No realistic docking ensemble, independent protein family test, charge sensitivity, calibration or external benchmark was performed. The output probability is an uncalibrated model score and should be used for ranking only.

## Next scientific step

A useful next validation set would comprise independently selected protein families and realistic docking decoys, with strict family or sequence-cluster held-out splits and native-pose redundancy checks. Pre-register descriptors and model settings before evaluating those structures; report failure cases, uncertainty, and geometry-only or clash-only baselines. For the electrostatics direction curate cofactors and protonation states, assign charges/radii, compute Poisson–Boltzmann potentials, and test grid and salt sensitivity. The exact commands and hashes for this release are in the [reproducibility record](REPRODUCIBILITY.md).
