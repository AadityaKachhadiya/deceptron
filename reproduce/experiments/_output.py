"""Prepare a separate output directory for a benchmark run."""

import argparse
import os
from pathlib import Path


def prepare_output(experiment):
    parser = argparse.ArgumentParser(description=f"Run the seed-7 {experiment} benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / experiment,
        help=f"New directory for artifacts (default: runs/{experiment}).",
    )
    output_dir = parser.parse_args().output_dir.resolve()
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        parser.error(f"Output directory already exists. Choose a new path: {output_dir}")
    print(f"Writing benchmark artifacts to {output_dir}", flush=True)
    os.chdir(output_dir)
