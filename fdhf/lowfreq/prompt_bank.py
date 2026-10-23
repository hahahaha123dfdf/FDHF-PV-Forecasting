"""Prompt-as-Prefix (PaP) prompt bank for the low-frequency Time-LLM branch.

The PaP prefix follows the template shown in the manuscript Fig 4 (the PaP
example from Jin et al., ICLR 2024), with four labeled sections:

  [Dataset Information]:      domain background (PV station, 15-min resolution)
  [Begin Data]:               ... [End Data] wrapper around the task + stats
  [Farm Type]:                station / farm type
  [Prediction Task]:          next <H> steps given the past <T> steps
  [Statistical Information]:  min / max / median / mean / trend (categorical)
                              / lag value (previous day, lag-96)

The prompt is tokenized and embedded by the frozen GPT-2 and prepended to the
reprogrammed time-series tokens (see time_llm.py).
"""

from __future__ import annotations


def build_pap_prompt(
    dataset: str = "pv",
    stats: dict | None = None,
    pred_len: int = 24,
    input_len: int = 96,
) -> str:
    """Build a structured natural-language PaP prefix (manuscript Fig 4 template).

    Parameters
    ----------
    dataset : str
        Key selecting the dataset description ("pv" supported).
    stats : dict, optional
        Per-window statistics from :func:`compute_window_stats`. When None, a
        generic (non-instance-specific) description is returned.
    pred_len, input_len : int
        Horizon <H> and lookback <T> written into the [Prediction Task] field
        (manuscript Section 3.2: next 24 steps based on the past 96 steps).
    """
    if dataset == "pv":
        desc = (
            "The dataset is 15-minute resolution photovoltaic power output of a "
            "20 MW station in Henan, China, driven by solar irradiance and "
            "weather. The low-frequency component captures the smooth daily "
            "trend and periodicity of solar generation."
        )
    else:
        desc = "A multivariate time series forecasting dataset."

    farm_type = "photovoltaic (PV) power station"
    task = (
        f"predict the next {pred_len} steps ({pred_len * 15 // 60} hours) "
        f"given the past {input_len} steps ({input_len * 15 // 60} hours)"
    )

    if stats is not None:
        s = stats
        stat_str = (
            f"minimum value {s.get('min', 0.0):.3f}, "
            f"maximum value {s.get('max', 0.0):.3f}, "
            f"median value {s.get('median', 0.0):.3f}, "
            f"mean value {s.get('mean', 0.0):.3f}, "
            f"trend {s.get('trend_cat', 'stable')}, "
            f"previous-day (lag-{input_len}) value {s.get('lag_val', 0.0):.3f}."
        )
        body = (
            f"[Farm Type]: {farm_type}\n"
            f"[Prediction Task]: {task}\n"
            f"[Statistical Information]:\n{stat_str}"
        )
    else:
        body = f"[Farm Type]: {farm_type}\n[Prediction Task]: {task}"

    # [Begin Data] ... [End Data] wrapper, exactly as in the Fig 4 template.
    prompt = f"[Dataset Information]:\n{desc}\n\n[Begin Data]:\n{body}\n[End Data]"
    return prompt


def compute_window_stats(window: "np.ndarray", input_len: int = 96) -> dict:
    """Compute the statistics written into the [Statistical Information] field.

    Fields of the PaP template (manuscript Fig 4): min, max,
    median, mean, categorical trend (upward / downward / stable) and the
    previous-day (lag-96 for 15-min data) value. The numeric slope and lag-1
    autocorrelation are kept as extra fields for potential ablation use.
    """
    import numpy as np

    w = window.astype(float)
    if w.ndim > 1:
        w = w.reshape(-1)
    mean = float(np.mean(w))
    std = float(np.std(w))
    median = float(np.median(w))
    vmin = float(np.min(w))
    vmax = float(np.max(w))
    if len(w) > 1:
        slope = float(w[-1] - w[0]) / max(len(w) - 1, 1)
        acf1 = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if np.std(w[:-1]) > 0 else 0.0
    else:
        slope, acf1 = 0.0, 0.0
    # Categorical trend, as in the PaP template ("upward / downward / stable").
    eps = 0.1 * std if std > 0 else 1e-8
    if slope > eps:
        trend_cat = "upward"
    elif slope < -eps:
        trend_cat = "downward"
    else:
        trend_cat = "stable"
    # Previous-day value: the observation one seasonal period (24 h = 96 steps
    # for 15-min data) earlier, i.e. the first element of a full-day window.
    lag_val = float(w[-input_len]) if len(w) >= input_len else float(w[0])
    return {
        "min": vmin,
        "max": vmax,
        "median": median,
        "mean": mean,
        "std": std,
        "trend": slope,
        "trend_cat": trend_cat,
        "acf1": acf1,
        "lag_val": lag_val,
    }
