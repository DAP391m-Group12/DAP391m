"""Compatibility alias for canonical RQ4a ownership."""

import sys
from importlib import import_module

_CANONICAL = import_module("src.dataset_b.research.rq4.rq4a.metrics")
sys.modules[__name__] = _CANONICAL
