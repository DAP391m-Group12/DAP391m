"""Dataset B navigation package.

Importing this package is intentionally CPU-safe and does not import model,
training, evaluation, checkpoint, or notebook execution code.
"""

from .core.config import ConfigResolutionError, config_path

__all__ = ["ConfigResolutionError", "config_path"]
