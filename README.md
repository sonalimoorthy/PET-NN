# PET-NN

A JAX-based neural network surrogate for **Physiological Equivalent Temperature (PET)** prediction. It replaces the iterative `pythermalcomfort.pet_steady` solver with a lightweight neural network that produces predictions in milliseconds, at ~0.05–0.06 °C MAE, for large-scale urban climate simulations and thermal comfort studies.

---

## Overview

PET is a widely used thermal comfort index that quantifies how humans perceive outdoor environmental conditions. Computing it via `pet_steady` requires an iterative thermophysiological heat-balance solve, which is too slow to run at scale (e.g. per grid cell, per timestep, across a city). This repository trains an MLP surrogate on `pet_steady`-labelled data that reproduces the solver's output to within ~0.05–0.06 °C, at over 1000× the speed.

The repo contains both the original exploratory pipeline (**`PET_Solver.ipynb`**) and a refactored, standalone version split into scripts:

* **`data_generation.py`** — generates `data/train.csv` and `data/test.csv`
* **`method1.py`** — MLP surrogate with plain `/100`-style input scaling
* **`method2.py`** — MLP surrogate with mean-centered input scaling

Unlike the notebook, the scripts do scaling *inside* the model (raw physical units go in, PET in °C comes out — no external `scale_X`/`scale_Y`/`unscale_Y` step).

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

Inputs are sampled uniformly at random over the ranges above. Near the edges of the box (high `tdb` combined with high `rh`), the underlying heat-balance solver can fail to converge and raises a `RuntimeWarning`. Training/test data is restricted to a **safe zone**: at or below `tdb=35°C` any humidity is allowed, and above 35 °C the maximum allowed humidity decreases linearly with temperature down to 50% by 50 °C. Any sample that still triggers a solver `RuntimeWarning` is discarded and resampled from scratch, so the shipped datasets have a 0% warning rate.

The training set additionally uses **hard-region oversampling**: `tdb > 35 °C` is where the trained surrogates' error is highest, and the safe-zone rule above already makes that region sample-sparser (lower accept rate). After drawing a uniform 30,000-point base set, `data_generation.py` measures that region's samples-per-degree and draws extra `tdb ∈ [35, 50]`-only points to top it up to match the rest of the range (currently ~5,000 extra points, for ~35,000 total training rows). The 10,000-point **test set stays uniform** (drawn from the same random stream position the original baseline used) so accuracy numbers reflect true uniform-deployment conditions, not the boosted training distribution.

---

## Model Architecture

A two-branch MLP implemented in Flax/JAX: a **temperature branch** (`tdb`, `tr`) and an **air branch** (`v`, `rh`, plus an engineered `log_v` feature), which merge before the output head.

```text
tdb, tr ──► Dense(128) ─ SiLU ─► Dense(256) ─ SiLU ─┐
                                                     ├─► concat ─► Dense(256) ─ SiLU ─► Dense(256) ─ SiLU ─► Dense(1) ─► PET
v, rh, log_v ──► Dense(256) ─ SiLU ─► Dense(256) ─ SiLU ─┘
```

* **Activation:** `SiLU`, with standard He-normal weight init.
* **Feature scaling** differs between the two script variants (both keep the same `log_v = (log(v) - log(V_MIN)) / (log(V_MAX) - log(V_MIN))` wind feature, since `v` spans 4 orders of magnitude and a linear scaling alone leaves almost no resolution in the low-wind regime):
  * **method1** (`/100`-style): `tdb/100`, `tr/100`, `v/10`, `rh/100`
  * **method2** (mean-centered): `(tdb-30)/100`, `(tr-50)/100`, `v/10`, `(rh-80)/100`
* Output `pet` is scaled by `/100` and unscaled by `×100` internally — both scripts return PET directly in °C from raw inputs.

**Why `SiLU` and not a periodic (SIREN-style) activation:** an earlier version of this model used `sin(2π·x)` activations (a SIREN-style design intended for high-frequency implicit signal fitting, e.g. images/audio). PET is a smooth, low-frequency function of 4 physical inputs, and stacking that periodic nonlinearity 4 layers deep produced a rugged loss landscape the optimizer couldn't escape — training converged to R² ≈ 0 (MAE ≈ 11 °C), i.e. it just predicted the mean. Switching to `SiLU` + He init on the same two-branch structure fixed convergence entirely.

### Training

* **Framework:** JAX + Flax + Optax
* **Optimizer:** Adam with a warmup–cosine-decay learning rate schedule (peak 1e-3, decayed to 1e-6 over the full run)
* **Loss:** MSE on raw-°C PET, tracked alongside held-out validation MAE
* **Epochs:** 1200 (full budget, no early stopping)
* **Batch size:** 512
* **Training loop:** data is moved to device once and each epoch runs as a single `jax.lax.scan` call over pre-batched, pre-shuffled data, instead of a Python loop dispatching one JIT call per batch. This was the main lever for CPU training speed — the scanned version trains ~1200 epochs in well under 15 minutes on CPU alone.

---

## Performance

Benchmarked on the 10,000-sample uniform held-out test set after the full 1200-epoch run, with hard-region oversampling applied to the training set:

| Metric                         | method1     | method2     |
| ------------------------------- | ----------- | ----------- |
| Test MAE                        | 0.0539 °C   | 0.0636 °C   |
| Test RMSE                       | 0.0848 °C   | 0.0938 °C   |
| Max absolute error              | 1.620 °C    | 1.589 °C    |
| Relative L2 error               | 0.244%      | 0.270%      |
| Mean absolute relative error    | 0.155%      | 0.197%      |
| NN inference, 10k samples       | ~0.04 s     | ~0.03 s     |
| **Speed-up vs `pet_steady`**    | **>1000×**  | **>1000×**  |

Full per-run metrics (including R², reference-solver timing) are in `output/benchmark_method1.csv` and `output/benchmark_method2.csv`; per-sample predictions and errors are in `output/predictions_method1.csv` / `output/predictions_method2.csv`.

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

(`jupyter` is only needed if you plan to run `PET_Solver.ipynb`; the standalone scripts below don't require it.)

---

## Usage

### Standalone scripts

```bash
python data_generation.py   # writes data/train.csv, data/test.csv
python method1.py           # trains + benchmarks the /100-scaling variant
python method2.py           # trains + benchmarks the mean-centered variant
```

Each training script saves its trained params (`output/model_method*.pickle`), a benchmark CSV, a predictions CSV, and a parity plot.

### Notebook (original, exploratory pipeline)

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
├── PET_Solver.ipynb              # Original notebook: data gen, model, training, benchmarking, boundary-case analysis
├── PET_Solver_Explained_1.html   # Rendered walkthrough of the notebook
├── data_generation.py            # Standalone data generation (safe-zone + hard-region oversampling)
├── method1.py                    # MLP surrogate, /100-style scaling
├── method2.py                    # MLP surrogate, mean-centered scaling
├── data/                         # Generated train/test CSVs + distribution plots
├── output/                       # Trained model params, benchmarks, predictions, parity plots
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
* Max absolute error (~1.6 °C) is driven by a small number of individually hard points where `pet_steady`'s response surface is sharply nonlinear — more training data alone doesn't fully resolve these.

---

## Future Improvements

* Extend to variable physiological parameters (age, sex, clothing, metabolic rate) as additional inputs
* Uncertainty quantification
* GPU-optimized training/deployment pipeline
* Seed ensembling to further reduce worst-case error

---

## Citation

If you use this repository in your research, please cite the associated project or publication when available.
