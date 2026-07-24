# PET-NN

A JAX-based neural network surrogate for **Physiological Equivalent Temperature (PET)** prediction. It replaces the iterative `pythermalcomfort.pet_steady` solver with a lightweight neural network that produces predictions in milliseconds, at ~0.05 °C accuracy, for large-scale urban climate simulations and thermal comfort studies.

---

## Overview

PET is a widely used thermal comfort index that quantifies how humans perceive outdoor environmental conditions. Computing it via `pet_steady` requires an iterative thermophysiological heat-balance solve, which is too slow to run at scale (e.g. per grid cell, per timestep, across a city). This repository trains an MLP surrogate on `pet_steady`-labelled data that reproduces the solver's output to within ~0.05 °C, at over 600× the speed.

The full pipeline — data generation, model, training, and benchmarking — lives in **`PET_Solver.ipynb`**.

---

## Input Variables

The model predicts PET from 4 variable environmental inputs:

| Variable | Description               | Unit | Range used for training |
| -------- | -------------------------- | ---- | ------------------------ |
| `tdb`    | Dry-bulb air temperature   | °C   | 10 – 50                  |
| `tr`     | Mean radiant temperature   | °C   | 10 – 80                  |
| `v`      | Wind speed                 | m/s  | 0.0003 – 10              |
| `rh`     | Relative humidity          | %    | 30 – 100                 |

All other `pet_steady` inputs are held fixed: metabolic rate (MET=1.7), clothing insulation (CLO=0.45), age (35), sex (male), height (1.7 m), weight (65 kg), atmospheric pressure (1010 hPa), sitting posture, and zero external work.

### Data generation

Inputs are sampled uniformly at random over the ranges above. Near the edges of the box (high `tdb` combined with high `rh`), the underlying heat-balance solver can fail to converge and raises a `RuntimeWarning`. Training data is restricted to a **safe zone**: at or below 30 °C any humidity is allowed, and above 30 °C the maximum allowed humidity decreases piecewise-linearly with temperature (down to 40% by 46 °C+). This keeps the solver's `RuntimeWarning` rate at 0% in the training/test sets, versus ~4.9% for unrestricted uniform sampling. The notebook's boundary-case analysis section compares this safe-zone restriction against plain unrestricted sampling and a resample-on-warning strategy across 30,000-sample draws.

30,000 training samples and 10,000 test samples are generated this way and labelled with the real `pet_steady` solver.

---

## Model Architecture

A two-branch MLP implemented in Flax/JAX: a **temperature branch** (`tdb`, `tr`) and an **air branch** (`v`, `rh`, plus an engineered `log_v` feature), which merge before the output head.

```text
tdb, tr ──► Dense(128) ─ SiLU ─► Dense(256) ─ SiLU ─┐
                                                     ├─► concat ─► Dense(256) ─ SiLU ─► Dense(256) ─ SiLU ─► Dense(1) ─► PET
v, rh, log_v ──► Dense(256) ─ SiLU ─► Dense(256) ─ SiLU ─┘
```

* **Activation:** `SiLU`, with standard He-normal weight init.
* **Feature scaling:** `tdb/100`, `tr/100`, `v/10`, `rh/100`, plus `log_v = (log(v) - log(V_MIN)) / (log(V_MAX) - log(V_MIN))`. The extra log-scaled wind feature exists because `v` spans 4 orders of magnitude (0.0003–10 m/s); a linear scaling alone leaves almost no resolution in the low-wind regime. Output `pet` is scaled by `/100` and unscaled by `×100`.

**Why `SiLU` and not a periodic (SIREN-style) activation:** an earlier version of this model used `sin(2π·x)` activations (a SIREN-style design intended for high-frequency implicit signal fitting, e.g. images/audio). PET is a smooth, low-frequency function of 4 physical inputs, and stacking that periodic nonlinearity 4 layers deep produced a rugged loss landscape the optimizer couldn't escape — training converged to R² ≈ 0 (MAE ≈ 11 °C), i.e. it just predicted the mean. Switching to `SiLU` + He init on the same two-branch structure fixed convergence entirely.

### Training

* **Framework:** JAX + Flax + Optax
* **Optimizer:** Adam with a warmup–cosine-decay learning rate schedule (peak 1e-3, decayed to 1e-6 over the full run)
* **Loss:** MSE on scaled PET, tracked alongside held-out validation MAE in real °C
* **Epochs:** 1200 (full budget, no early stopping)
* **Batch size:** 512
* **Training loop:** data is moved to device once and each epoch runs as a single `jax.lax.scan` call over pre-batched, pre-shuffled data, instead of a Python loop dispatching one JIT call per batch. This was the main lever for CPU training speed — the scanned version trains ~1200 epochs in well under 15 minutes on CPU alone.

---

## Performance

Benchmarked on the 10,000-sample held-out test set after the full 1200-epoch run:

| Metric                         | Value      |
| ------------------------------- | ---------- |
| Validation MAE                  | 0.0482 °C  |
| Test MAE                        | 0.0484 °C  |
| Test RMSE                       | 0.089 °C   |
| Test R²                         | 0.99995    |
| Max absolute error              | 4.72 °C    |
| Relative L2 error               | 0.29%      |
| Mean absolute relative error    | 0.16%      |
| NN inference, 10k samples       | ~0.09 s    |
| `pet_steady`, 10k samples       | ~64 s      |
| **Speed-up**                    | **~684×**  |

---

## Installation

```bash
git clone https://github.com/sonalimoorthy/PET-NN.git
cd PET-NN
```

Create an environment and install dependencies:

```bash
conda create -n pet_nn python=3.11
conda activate pet_nn
pip install jax flax optax scikit-learn numpy pandas matplotlib seaborn pythermalcomfort jupyter
```

---

## Usage

Launch the notebook:

```bash
jupyter notebook PET_Solver.ipynb
```

It includes, in order: reference-solver wrapper, safe-zone data generation, feature scaling, model definition, training, benchmarking against `pet_steady`, and a boundary-case analysis comparing sampling strategies.

### Inference example

```python
tdb, tr, v, rh = 30.0, 40.0, 1.0, 50.0
pet_nn(tdb, tr, v, rh)   # -> PET in °C, from the trained surrogate
```

Input order is `(tdb, tr, v, rh)`; `pet_nn` also accepts arrays for batched inference.

---

## Repository Structure

```text
PET-NN/
│
├── PET_Solver.ipynb   # Data generation, model, training, benchmarking, boundary-case analysis
└── README.md
```

---

## Applications

* Urban climate simulations
* Thermal comfort assessment
* Urban digital twins
* Building performance analysis
* Smart city applications
* Environmental modelling

---

## Limitations

* Trained only over `tdb` 10–50 °C, `tr` 10–80 °C, `v` 0.0003–10 m/s, `rh` 30–100% (and further restricted to the safe zone at high temperature/humidity) — accuracy outside this range is untested.
* All non-variable `pet_steady` inputs (metabolic rate, clothing, age, sex, body size, etc.) are fixed at the values above; the surrogate does not generalize across different physiological parameters without retraining.
* The training data itself is only as accurate as `pet_steady`.

---

## Future Improvements

* Extend to variable physiological parameters (age, sex, clothing, metabolic rate) as additional inputs
* Uncertainty quantification
* GPU-optimized training/deployment pipeline

---

## Citation

If you use this repository in your research, please cite the associated project or publication when available.
