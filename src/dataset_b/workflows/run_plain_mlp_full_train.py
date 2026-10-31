"""Canonical Command-line entry point for the PMLP-F04 full-Train stage."""

import sys

from src.dataset_b.models.neural.plain_mlp.training import main


if __name__ == "__main__":
    raise SystemExit(main(["full-train", *sys.argv[1:]]))
