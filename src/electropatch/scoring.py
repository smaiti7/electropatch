"""Four predefined interface descriptors and one fixed ranking formula."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .structures import Atoms


@dataclass(frozen=True)
class Features:
    contacts: int
    clashes: int
    basic_phosphate_contacts: int
    mean_nearest_distance: float


def calculate_features(protein: Atoms, dna: Atoms, dna_coords: np.ndarray) -> Features:
    """Compute heavy-atom features for one pose, in ångström units.

    Mean nearest distance uses ALL DNA heavy atoms so its definition is fixed
    across poses, including poses with no interface contacts.
    """
    protein_tree = cKDTree(protein.coords)
    dna_tree = cKDTree(dna_coords)
    contacts = sum(len(neighbors) for neighbors in protein_tree.query_ball_tree(dna_tree, 4.5))
    clashes = sum(len(neighbors) for neighbors in protein_tree.query_ball_tree(dna_tree, 2.0))
    basic_indices = [
        i for i, (name, residue) in enumerate(zip(protein.names, protein.residues))
        if (residue == "LYS" and name == "NZ")
        or (residue == "ARG" and name in {"NE", "NH1", "NH2"})
    ]
    phosphate_indices = [
        i for i, name in enumerate(dna.names) if name in {"OP1", "OP2", "O1P", "O2P"}
    ]
    if basic_indices and phosphate_indices:
        basic_tree = cKDTree(protein.coords[basic_indices])
        phosphate_tree = cKDTree(dna_coords[phosphate_indices])
        basic_phosphate_contacts = sum(
            len(neighbors) for neighbors in basic_tree.query_ball_tree(phosphate_tree, 4.5)
        )
    else:
        basic_phosphate_contacts = 0
    nearest, _ = protein_tree.query(dna_coords)
    return Features(contacts, clashes, basic_phosphate_contacts, float(nearest.mean()))


def score_variants(features: list[Features]) -> tuple[np.ndarray, np.ndarray]:
    """Return geometry and predefined combined scores for a fixed pose pool."""
    data = np.array(
        [[f.contacts, f.clashes, f.basic_phosphate_contacts, f.mean_nearest_distance] for f in features],
        dtype=float,
    )
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=0)
    z = np.divide(data - mean, std, out=np.zeros_like(data), where=std > 0)
    geometry = z[:, 0] - 2*z[:, 1] - z[:, 3]
    combined = geometry + z[:, 2]
    return geometry, combined


def combined_scores(features: list[Features]) -> np.ndarray:
    """Apply the predeclared score with population z-scores across all poses."""
    return score_variants(features)[1]
