"""Compatibility alias for the canonical Dataset B split-input boundary."""

import sys
from importlib import import_module

_CANONICAL = import_module("src.dataset_b.data.split_inputs")
sys.modules[__name__] = _CANONICAL
