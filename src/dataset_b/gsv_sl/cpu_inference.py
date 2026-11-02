"""Compatibility alias for canonical research ownership."""

import sys
from importlib import import_module

_CANONICAL = import_module("src.dataset_b.research.rq2.cpu_inference")
sys.modules[__name__] = _CANONICAL
