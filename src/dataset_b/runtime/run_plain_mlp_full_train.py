"""Command-line entry point for the PMLP-F04 full-Train stage."""

import sys

from .plain_mlp_training import main


if __name__ == "__main__":
    raise SystemExit(main(["full-train", *sys.argv[1:]]))
