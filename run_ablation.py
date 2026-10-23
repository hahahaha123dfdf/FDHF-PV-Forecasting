"""Run the ablation study (manuscript Experiment 4, Table 8) with seeds and
Wilcoxon significance tests.

For each case the pipeline removes/changes exactly one module while keeping
everything else identical, then reports MAE / RMSE / R2 on the test set as
mean +/- std over the configured seeds. A Wilcoxon signed-rank test (point-wise
absolute errors, full model as reference; manuscript Section 4.2) is reported
for every case against the Full model.

Usage
-----
    python run_ablation.py --config configs/default.yaml --data path/to/pv_scada.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from fdhf.pipeline import ABLATIONS, FDHFPipeline
from fdhf.data.metrics import evaluate_with_std, wilcoxon_signed_rank


def main():
    ap = argparse.ArgumentParser(description="FDHF ablation study.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--capacity", type=float, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.data:
        cfg["data"]["path"] = args.data
    seeds = cfg["training"]["seeds"]

    cases = list(ABLATIONS.keys())  # [None, "1a", "1b", "2a", "2b", "2c", "3", "4"]
    per_case = {}        # case -> list of metric dicts
    per_case_err = {}    # case -> point-wise absolute error array (last seed)
    print(f"{'Case':<8}{'Setting':<22}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for case in cases:
        m_list, errs = [], None
        for seed in seeds:
            cfg["training"]["seed"] = seed
            pipe = FDHFPipeline(cfg, experiment="full", ablation=case)
            m = pipe.run(cfg["data"]["path"], capacity_mw=args.capacity, seed=seed)
            m_list.append(m)
            errs = np.abs(pipe.test_y_true - pipe.test_y_pred)
        per_case[case] = m_list
        per_case_err[case] = errs
        agg = evaluate_with_std(m_list)
        label = "Full" if case is None else f"w/o {case}"
        print(f"{str(case):<8}{label:<22}"
              f"{agg['MAE'][0]:>10.4f}{agg['RMSE'][0]:>10.4f}{agg['R2'][0]:>10.4f}")

    # Wilcoxon: each case vs Full (reference)
    full_err = per_case_err[None]
    print("\nWilcoxon signed-rank test (vs Full, last-seed point-wise abs errors):")
    for case in cases:
        if case is None:
            continue
        w = wilcoxon_signed_rank(full_err, per_case_err[case])
        p = w.get("p")
        ps = f"{p:.4f}" if p is not None else "n/a (scipy missing)"
        print(f"  w/o {case:<5} -> statistic={w['statistic']:.3f}, p={ps}")


if __name__ == "__main__":
    main()
