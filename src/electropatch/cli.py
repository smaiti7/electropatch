"""Command-line interface for single-complex and comparative experiments."""

import argparse
from pathlib import Path

from .comparison import run_comparison
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="electropatch", description="Compare protein-DNA native poses with rigid-body decoys")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("run", help="generate decoys, score poses, and write figures")
    command.add_argument("--pdb-file", type=Path, default=Path("data/1AAY.pdb"))
    command.add_argument("--output-dir", type=Path, default=Path("."))
    command.add_argument("--protein", default="A", help="PDB author chain ID")
    command.add_argument("--dna", default="B,C", help="comma-separated PDB author chain IDs")
    command.add_argument("--n-decoys", type=int, default=50, help="positive even number")
    command.add_argument("--seed", type=int, default=42)
    comparison = subparsers.add_parser("compare", help="run every checksum-verified complex in the manifest")
    comparison.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    comparison.add_argument("--output-dir", type=Path, default=Path("."))
    comparison.add_argument("--n-decoys", type=int, default=50, help="positive even number")
    comparison.add_argument("--seed", type=int, default=42)
    ml = subparsers.add_parser("ml", help="leave-one-complex-out logistic ranking on existing comparison results")
    ml.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ml.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "run":
        summary = run(args.pdb_file, args.output_dir, tuple(x.strip() for x in args.protein.split(",")),
                      tuple(x.strip() for x in args.dna.split(",")), args.n_decoys, args.seed)
        print(f"Native rank: {summary['native_rank']} of {summary['n_poses']}")
        print(f"Results: {args.output_dir / 'results' / 'pose_scores.csv'}")
        print(f"Figures: {args.output_dir / 'figures'}")
    elif args.command == "compare":
        rows = run_comparison(args.manifest, args.output_dir, args.n_decoys, args.seed)
        for row in rows:
            print(f"{row['pdb_id']}: geometry {row['native_rank_geometry']}/{row['n_poses']}; "
                  f"combined {row['native_rank_combined']}/{row['n_poses']}")
        print(f"Comparison: {args.output_dir / 'results' / 'comparison.csv'}")
    elif args.command == "ml":
        from .ml import run_ml
        rows = run_ml(args.manifest, args.output_dir)
        for row in rows:
            print(f"{row['pdb_id']}: held-out ML native rank {row['native_rank_ml']}/{row['n_poses']}")
        print(f"ML comparison: {args.output_dir / 'results' / 'ml_comparison.csv'}")


if __name__ == "__main__":
    main()
