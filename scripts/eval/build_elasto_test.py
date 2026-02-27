#!/usr/bin/env python3
"""
Build expanded elastoplastic test set from full data,
excluding simulations already in train/val/test splits.

Usage:
    python build_elasto_test.py \
        --full_dir /path/to/all_simulations \
        --small_dir /path/to/small (contains train/ val/ test/) \
        --output_dir /path/to/expanded_test
"""

import argparse
import shutil
from pathlib import Path


def get_sim_names(directory):
    """Get set of simulation stem names from a directory of .pt files."""
    d = Path(directory)
    if not d.exists():
        return set()
    return {f.stem for f in d.glob("simulation_*.pt")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", required=True,
                        help="Directory with ALL simulation .pt files")
    parser.add_argument("--small_dir", required=True,
                        help="Directory with train/ val/ test/ subdirs to exclude")
    parser.add_argument("--output_dir", required=True,
                        help="Where to symlink/copy the expanded test set")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking (uses more disk)")
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    small_dir = Path(args.small_dir)
    output_dir = Path(args.output_dir)

    # Collect all names to exclude
    exclude = set()
    for split in ['train', 'val', 'test']:
        names = get_sim_names(small_dir / split)
        print(f"  {split}: {len(names)} simulations")
        exclude |= names

    print(f"  Total excluded: {len(exclude)}")

    # Get all available simulations (check subdirs and root)
    all_files = sorted(full_dir.glob("simulation_*.pt"))
    for subdir in ['train', 'test', 'val']:
        all_files.extend(sorted((full_dir / subdir).glob("simulation_*.pt")))
    print(f"  Full dataset: {len(all_files)} simulations")

    # Filter
    eligible = [f for f in all_files if f.stem not in exclude]
    print(f"  Eligible for testing: {len(eligible)}")

    # Create output
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in eligible:
        dst = output_dir / f.name
        if dst.exists():
            continue
        if args.copy:
            shutil.copy2(f, dst)
        else:
            dst.symlink_to(f.resolve())

    final_count = len(list(output_dir.glob("simulation_*.pt")))
    print(f"\n  ✓ {final_count} simulations in {output_dir}")


if __name__ == "__main__":
    main()