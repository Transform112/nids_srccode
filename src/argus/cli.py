"""Unified ARGUS CLI entry point — thin dispatcher to the numbered scripts in
`scripts/`, which are the primary, independently-runnable pipeline stages.

Usage:
    argus prepare-data --dataset cicids2018 --nrows 200000
    argus build-splits --dataset cicids2018 --protocol A
    argus fit-features --dataset cicids2018
    argus train-stage1 --dataset cicids2018
    argus train-stage2 --dataset cicids2018
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[2] / "scripts"

COMMAND_SCRIPT = {
    "prepare-data": "01_prepare_data.py",
    "build-splits": "02_build_splits.py",
    "fit-features": "03_fit_features.py",
    "train-stage1": "04_train_encoder.py",
    "train-stage2": "05_train_head.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="argus")
    parser.add_argument("command", choices=sorted(COMMAND_SCRIPT))
    args, rest = parser.parse_known_args(argv)

    script = SCRIPTS_DIR / COMMAND_SCRIPT[args.command]
    result = subprocess.run([sys.executable, str(script), *rest])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

