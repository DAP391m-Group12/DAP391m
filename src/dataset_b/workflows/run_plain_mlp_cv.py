"""Canonical Command-line entry point for PMLP-F04 CV."""

import sys

from src.dataset_b.models.neural.plain_mlp.training import main


if __name__ == "__main__":
    raise SystemExit(main(["cv", *sys.argv[1:]]))
