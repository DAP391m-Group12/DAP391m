"""Compatibility alias for canonical RQ1 reporting."""

import sys
from importlib import import_module

_CANONICAL = import_module("src.dataset_b.research.rq1.reporting")
sys.modules[__name__] = _CANONICAL
