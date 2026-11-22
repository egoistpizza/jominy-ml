"""Command line entry point: ``jominy run`` reproduces every result in results/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jominy.latex import Outputs
from jominy.pipeline import run as run_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jominy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the full pipeline and write results/")
    run.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.command == "run":
        summary = run_pipeline(Outputs(args.out))
        with pd.option_context("display.width", 140, "display.precision", 2):
            print(summary[["mae", "mae_low", "mae_high", "rmse", "bias"]])


if __name__ == "__main__":
    main()
