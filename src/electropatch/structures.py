"""Minimal PDB ATOM reader with explicit chain and alternate-location handling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Atoms:
    coords: np.ndarray
    names: tuple[str, ...]
    residues: tuple[str, ...]
    chains: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.names)


def read_heavy_atoms(path: str | Path, chains: tuple[str, ...]) -> Atoms:
    """Read first-model, ATOM-only heavy atoms for author PDB chain IDs.

    For duplicate alternate locations, retain the highest-occupancy record,
    breaking equal-occupancy ties in favor of altloc A, then blank.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDB file does not exist: {path}")
    wanted = set(chains)
    selected: dict[tuple[str, str, str, str], tuple[float, int, str, str, np.ndarray]] = {}
    saw_model = False
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MODEL "):
                if saw_model:
                    break
                saw_model = True
                continue
            if line.startswith("ENDMDL") and saw_model:
                break
            if not line.startswith("ATOM  "):
                continue
            chain = line[21:22]
            if chain not in wanted:
                continue
            name = line[12:16].strip()
            element = line[76:78].strip().upper() or name.lstrip("0123456789")[:1].upper()
            if element in {"H", "D"}:
                continue
            try:
                xyz = np.array([float(line[i : i + 8]) for i in (30, 38, 46)], dtype=float)
                occupancy = float(line[54:60]) if line[54:60].strip() else 0.0
            except ValueError as exc:
                raise ValueError(f"Malformed atom record in {path}: {line.rstrip()}") from exc
            residue = line[17:20].strip()
            altloc = line[16:17]
            key = (chain, line[22:26], line[26:27], name)
            priority = 2 if altloc == "A" else 1 if altloc == " " else 0
            previous = selected.get(key)
            if previous is None or (occupancy, priority) > (previous[0], previous[1]):
                selected[key] = (occupancy, priority, name, residue, xyz)
    observed = {key[0] for key in selected}
    missing = wanted - observed
    if missing:
        raise ValueError(f"Missing requested PDB chain(s): {', '.join(sorted(missing))}")
    items = list(selected.items())
    return Atoms(
        coords=np.stack([record[4] for _, record in items]),
        names=tuple(record[2] for _, record in items),
        residues=tuple(record[3] for _, record in items),
        chains=tuple(key[0] for key, _ in items),
    )


def load_complex(
    path: str | Path,
    protein: str | tuple[str, ...] = "A",
    dna: tuple[str, ...] = ("B", "C"),
) -> tuple[Atoms, Atoms]:
    protein_chains = (protein,) if isinstance(protein, str) else protein
    if (not protein_chains or not dna or
            any(len(chain) != 1 for chain in (*protein_chains, *dna))):
        raise ValueError("Use one-character PDB chain IDs and at least one chain per partner")
    if set(protein_chains) & set(dna) or len(set(protein_chains)) != len(protein_chains) or len(set(dna)) != len(dna):
        raise ValueError("Protein and DNA chain IDs must be distinct")
    return read_heavy_atoms(path, protein_chains), read_heavy_atoms(path, dna)
