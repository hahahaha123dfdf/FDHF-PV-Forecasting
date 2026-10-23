"""Entry point for the full FDHF experiment (manuscript Section 4).

Usage
-----
    python run_fdhf.py --config configs/default.yaml \
        --data path/to/pv_scada.csv --experiment full

Experiments: full | scarce15 | scarce30 | scarce45
Seeds: all seeds listed under training.seeds are run; metrics are reported as
mean +/- std (manuscript Section 4.2).
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from fdhf.pipeline import FDHFPipeline
from fdhf.data.metrics import evaluate_with_std


def main():
    ap = argparse.ArgumentParser(description="Run the FDHF forecasting framework.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default=None, help="Override CSV path in config.")
    ap.add_argument("--experiment", default="full",
                    choices=["full", "scarce15", "scarce30", "scarce45"])
    ap.add_argument("--capacity", type=float, default=None,
                    help="Rated capacity (MW) for outlier masking.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.data:
        cfg["data"]["path"] = args.data

    seeds = cfg["training"]["seeds"]
    results = []
    for seed in seeds:
        cfg["training"]["seed"] = seed
        pipe = FDHFPipeline(cfg, experiment=args.experiment, ablation=None)
        m = pipe.run(cfg["data"]["path"], capacity_mw=args.capacity, seed=seed)
        results.append(m)
        print(f"[seed {seed}] {args.experiment}: "
              f"MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")

    agg = evaluate_with_std(results)
    print(f"\n[{args.experiment}] mean +/- std over {len(seeds)} seeds:")
    for k in ("MAE", "RMSE", "R2"):
        mean, std = agg[k]
        print(f"  {k:>6}: {mean:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
