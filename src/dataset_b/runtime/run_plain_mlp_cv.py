"""Command-line entry point for PMLP-F04 CV."""

import sys

from .plain_mlp_training import main


if __name__ == "__main__":
    raise SystemExit(main(["cv", *sys.argv[1:]]))
