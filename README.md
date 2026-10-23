# FDHF — Frequency-Domain Decoupled Hybrid Forecasting

Reference implementation of the framework described in:

> *Few-shot power prediction for newly-built photovoltaic power stations: A new
> framework integrating LLM and data augmentation methods*
> *Applied Soft Computing*

The framework predicts PV power under data-scarce conditions by **decoupling the
signal in the frequency domain** and modeling each branch with a specialised
model, then fusing them with ridge regression.

```
原始SCADA → 清洗 → 时序切分
        → MVMD-EN 分解 (仅在训练子集拟合)
        → 归一化 (min/max, 仅训练子集)
        → 样本熵分频
        → 低频分支: Time-LLM (冻结 GPT-2)  ─┐
        → 高频分支: WGAN-GP + XGBoost       ├─ 岭回归融合 → PV 预测
        → 岭回归融合                       ┘
```

---

## 1. Data

The SCADA data is **not redistributed** in this repository. It is publicly
available from the Shandong University PV Energy System Optimization (PSEO)
platform:

> **Dataset source:** <https://energymeteo-pseo.sdu.edu.cn/>

Place the exported CSV at `data/pv_scada.csv`. The loader expects:

| Item | Value |
|---|---|
| Datetime column | `datetime` |
| Target column | `power` |
| Resolution | 15 min |
| Coverage | 2018-06-30 ~ 2019-06-30 (1 year) |
| Station | Henan, China (36.644°N, 113.642°E), 20 MW |
| Meteorological columns | `GHI`, `DNI`, `temperature`, `wind_speed`, `wind_direction`, `pressure`, `DHI`, `local_temp`, `local_pressure`, `local_wind_speed`, `local_wind_direction` |

The exact column list is configurable in `configs/default.yaml` (`data.feature_cols`).

**To verify the pipeline without the real data**, generate a synthetic set with the
same schema:

```bash
python make_dummy_data.py --out data/pv_scada.csv --n 30000
```

---

## 2. Install

```bash
pip install -r requirements.txt          # flexible minimum versions
pip install -r requirements-lock.txt     # exact versions used for the reported results
```

> The first run of the Time-LLM branch downloads GPT-2-Small (~500 MB) from
> HuggingFace into the local cache; no weights are stored in this repository.

**Hardware used in the manuscript:** NVIDIA RTX 3090 (24 GB), Python 3.12, PyTorch 2.4.
The code has also been verified on Python 3.9.13 / PyTorch 2.0.1 and on CPU.

---

## 3. Run

```bash
# Full-data experiment
python run_fdhf.py --data data/pv_scada.csv --experiment full

# Few-shot (15% / 30% / 45% of the training period)
python run_fdhf.py --data data/pv_scada.csv --experiment scarce15
python run_fdhf.py --data data/pv_scada.csv --experiment scarce30
python run_fdhf.py --data data/pv_scada.csv --experiment scarce45

# Ablation study
python run_ablation.py --data data/pv_scada.csv
```

### Reproducibility checks

```bash
python _smoke_test.py                                    # end-to-end pipeline check
python benchmark_cost.py --csv data/pv_scada.csv --batch-size 32   # training / inference / memory
python benchmark_baselines.py --windows 2048              # baseline cost comparison
```

### Configuration

All hyperparameters in `configs/default.yaml` are taken directly from the
manuscript:

| Component | Setting |
|---|---|
| MVMD | K = 6 modes, α = 2000, tolerance 1.0e-3, max 500 iterations |
| Elastic Net | α = 0.01, l1_ratio = 0.5 |
| Sample-entropy threshold | S = 0.13, m = 2, r = 0.2 |
| Time-LLM | GPT-2-Small (frozen), patch 16 / stride 8, 10 epochs, lr 1.0e-4, batch 32, PaP enabled |
| WGAN-GP | GRU hidden 64, latent 32, λ = 10, 5 critic steps, 500 epochs, batch 32 |
| XGBoost | n_estimators 500, max_depth 5, lr 0.05, subsample 0.8, colsample_bytree 0.8 |
| Ridge fusion | `sklearn.linear_model.Ridge` |
| Seeds | 2024, 2025, 2026, 2027, 2028 |

---

## 4. Leakage control

* Splits are strictly chronological (`fdhf/data/loader.py`): 70% training period,
  30% test period; the last 15% of the training period is the validation set and
  is used for hyperparameter and threshold selection only.
* MVMD-EN and the normalisation bounds are fit on the **actual training subset
  only** and then applied to the validation and test subsets.
* The test set is never used for model selection, threshold selection or
  hyperparameter tuning, and synthetic WGAN-GP samples are generated from the
  training subset alone, so no augmented sample can enter the test set.
* For stricter per-window causality, `MVMDEN.decompose_window` provides a causal
  decomposition that can replace `decompose_series` in the pipeline.

---

## 5. Components and provenance

| Module | Source | License |
|--------|--------|---------|
| `fdhf/decomposition/mvmd.py` | [Dmocrito/mvmd](https://github.com/Dmocrito/mvmd) (vendored) | MIT |
| `fdhf/lowfreq/time_llm.py` | Re-implementation of the [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM) architecture (ICLR 2024) | — |
| `fdhf/highfreq/wgan_gp.py` | WGAN-GP (Gulrajani et al., 2017) with a GRU generator/discriminator | — |
| `fdhf/highfreq/xgboost_model.py` | `xgboost.XGBRegressor` | Apache-2.0 |
| `fdhf/fusion/ridge_fusion.py` | `sklearn.linear_model.Ridge` | BSD-3 |
| Elastic Net screening | `sklearn.linear_model.ElasticNet` | BSD-3 |

See `LICENSE` for the third-party notices.

---

## 6. Citation

If you use this code, please cite the corresponding article:

```bibtex
@article{fdhf_pv_fewshot,
  title  = {Few-shot power prediction for newly-built photovoltaic power stations:
            A new framework integrating LLM and data augmentation methods},
  author = {},
  journal = {Applied Soft Computing},
  year   = {2026},
  note   = {Under review}
}
```

## 7. License

MIT — see `LICENSE`.
