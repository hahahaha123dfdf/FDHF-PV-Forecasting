"""Baseline registry for the manuscript comparison tables (Tables 2, 4, 6, 7,
9, 10). Offline-feasible models are imported eagerly; torch / API models are
documented but only instantiated on demand."""

from __future__ import annotations

from .base import BaseForecaster, make_pv_windows
from .statistical import Persistence, SeasonalNaive
from .augment import jittering, bootstrapping

__all__ = [
    "BaseForecaster", "make_pv_windows", "Persistence", "SeasonalNaive",
    "jittering", "bootstrapping",
]

# Lazily-available (require torch / API): import on demand via these helpers.
def get_torch_baselines():
    from . import ml
    from . import vmd_lstm
    from . import dcgan
    return ml, vmd_lstm, dcgan
