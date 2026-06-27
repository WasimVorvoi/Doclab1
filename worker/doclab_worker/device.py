"""Device resolution for deep-learning paths (image, text).

Apple Silicon gets MPS (Metal); everything else falls back to CPU. Tabular
(XGBoost) stays CPU and does not use this module.
"""

from __future__ import annotations


def resolve_device() -> str:
    """Return 'mps' on Apple Silicon with a working MPS build, else 'cpu'."""
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"
