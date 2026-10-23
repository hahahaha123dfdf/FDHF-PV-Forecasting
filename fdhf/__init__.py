"""FDHF: Frequency-Domain Decoupled Hybrid Forecasting framework.

Few-shot PV power prediction integrating LLM (Time-LLM) and data augmentation
(WGAN-GP + XGBoost), with MVMD-EN decomposition and ridge-regression fusion.
"""

from .pipeline import ABLATIONS

try:  # torch-based pipeline is optional at import time
    from .pipeline import FDHFPipeline
    __all__ = ["FDHFPipeline", "ABLATIONS"]
except ImportError:  # pragma: no cover - torch / transformers not installed
    __all__ = ["ABLATIONS"]

__version__ = "0.1.0"
