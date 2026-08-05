"""Generate the PET training/test data: safe-zone-restricted, resample-on-warning
sampling, labelled with the real pet_steady solver. The training set is a
uniform base plus a hard-region top-up (tdb > HARD_TDB_MIN, where error is
highest) so that region's sample density matches the rest of the range; the
test set stays uniform for an unbiased benchmark. Writes data/train.csv,
data/test.csv, and histogram grids of both datasets' distributions.

Run standalone:
    python data_generation.py
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pythermalcomfort.models import pet_steady

# --------------------------------------------------------------------------
# Fixed physiological parameters (the "standard person")
# --------------------------------------------------------------------------
MET = 1.7
CLO = 0.45
AGE = 35
SEX = "male"
HEIGHT_M = 1.7
WEIGHT_KG = 65.0
P_ATM_HPA = 1010.0
POSITION = "sitting"
WME = 0.0

# --------------------------------------------------------------------------
# Variable input ranges
# --------------------------------------------------------------------------
TDB_MIN, TDB_MAX = 10.0, 50.0
TR_MIN, TR_MAX = 10.0, 80.0
V_MIN, V_MAX = 0.0003, 10.0
RH_MIN, RH_MAX = 30.0, 100.0

N_TRAIN = 30_000
N_TEST = 10_000
SEED = 2025

DATA_DIR = "data"

# --------------------------------------------------------------------------
# Safe-zone boundary: rh unrestricted (<=100) up to tdb=35, then capped
# linearly down to 50% at tdb=50 (the line identified in the boundary-case
# frontier analysis). Any accepted sample that still makes pet_steady raise a
# RuntimeWarning is discarded and resampled from scratch.
# --------------------------------------------------------------------------
SAFE_TDB_BREAKPOINTS = [35.0, 50.0]
SAFE_RH_BREAKPOINTS = [100.0, 50.0]

# --------------------------------------------------------------------------
# Hard-region oversampling: past tdb=35 the rh cap above already thins out
# accepted draws (lower acceptance rate -> lower sample density), and that's
# also where the trained surrogates' error is highest. After drawing the
# uniform base training set, we top up tdb>HARD_TDB_MIN with extra
# draws so its samples-per-degree matches the rest of the range. Test set is
# left uniform (untouched) so benchmark metrics stay an unbiased read of
# true deployment error.
# --------------------------------------------------------------------------
HARD_TDB_MIN = SAFE_TDB_BREAKPOINTS[0]


def safe_rh_max(tdb):
    return np.interp(tdb, SAFE_TDB_BREAKPOINTS, SAFE_RH_BREAKPOINTS)


def pet_reference(tdb, tr, v, rh):
    """Ground-truth PET [degC] for one weather condition, all physiological
    params fixed."""
    try:
        res = pet_steady(tdb=tdb, tr=tr, v=v, rh=rh, met=MET, clo=CLO,
                          p_atm=P_ATM_HPA, position=POSITION, age=AGE, sex=SEX,
                          weight=WEIGHT_KG, height=HEIGHT_M, wme=WME)
    except (TypeError, ValueError):
        pos_code = 1 if POSITION == "sitting" else 2
        sex_code = 1 if SEX == "male" else 2
        res = pet_steady(tdb=tdb, tr=tr, v=v, rh=rh, met=MET, clo=CLO,
                          p_atm=P_ATM_HPA, position=pos_code, age=AGE, sex=sex_code,
                          weight=WEIGHT_KG, height=HEIGHT_M, wme=WME)
    return float(res.pet) if hasattr(res, "pet") else float(res)


def generate_dataset(n, rng, tdb_min=TDB_MIN, tdb_max=TDB_MAX):
    """Draw (tdb, tr, v, rh) within the safe-zone boundary and label with
    pet_reference; whenever the solver raises a RuntimeWarning, discard that
    draw and resample all 4 inputs (still respecting the boundary) until a
    warning-free label is obtained. tdb is drawn from [tdb_min, tdb_max],
    which defaults to the full range but can be narrowed to target a
    specific tdb band (see HARD_TDB_MIN oversampling in main())."""
    X = np.empty((n, 4))
    y = np.empty(n)
    n_draws = 0
    n_resampled = 0
    t0 = time.perf_counter()
    for i in range(n):
        while True:
            tdb = rng.uniform(tdb_min, tdb_max)
            tr = rng.uniform(TR_MIN, TR_MAX)
            v = rng.uniform(V_MIN, V_MAX)
            rh = rng.uniform(RH_MIN, RH_MAX)
            if rh > safe_rh_max(tdb):
                continue  # outside the safe-zone boundary, redraw
            n_draws += 1
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pet = pet_reference(tdb, tr, v, rh)
                warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
            if warned:
                n_resampled += 1
                continue  # solver didn't converge, resample
            break
        X[i] = [tdb, tr, v, rh]
        y[i] = pet
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n} labelled  ({time.perf_counter() - t0:.1f}s, "
                  f"{n_draws} draws inside boundary, {n_resampled} resampled)")
    print(f"  done: {n} labelled ({n_draws} draws inside boundary, {n_resampled} resampled away, "
          f"{time.perf_counter() - t0:.1f}s)")
    return X, y


def save_distribution_plot(df, title, path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 3))
    for ax, col in zip(axes, ["tdb", "tr", "v", "rh", "pet"]):
        ax.hist(df[col], bins=50)
        ax.set_title(col)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"saved {path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print("Generating + labelling training set (uniform base):")
    X_train_base, Y_train_base = generate_dataset(N_TRAIN, rng)

    # test set is drawn here (right after the base train set, same as before
    # hard-region oversampling existed) so it consumes the same rng stream
    # position as the original baseline run -- keeps it a like-for-like,
    # unbiased benchmark unaffected by the extra draws below.
    print("Generating + labelling test set (uniform, unchanged -- unbiased benchmark):")
    X_test_raw, Y_test_raw = generate_dataset(N_TEST, rng)

    n_hard_base = int((X_train_base[:, 0] > HARD_TDB_MIN).sum())
    easy_density = (N_TRAIN - n_hard_base) / (HARD_TDB_MIN - TDB_MIN)  # samples / degC
    target_hard_n = int(round(easy_density * (TDB_MAX - HARD_TDB_MIN)))
    n_hard_extra = max(0, target_hard_n - n_hard_base)
    print(f"  base set has {n_hard_base}/{N_TRAIN} samples with tdb>{HARD_TDB_MIN} "
          f"({n_hard_base / N_TRAIN:.1%}); topping up with {n_hard_extra} more there "
          f"to match the {easy_density:.0f}/degC density of tdb<={HARD_TDB_MIN}")

    print(f"Generating + labelling hard-region top-up (tdb in [{HARD_TDB_MIN}, {TDB_MAX}]):")
    X_train_hard, Y_train_hard = generate_dataset(
        n_hard_extra, rng, tdb_min=HARD_TDB_MIN, tdb_max=TDB_MAX)

    X_train_raw = np.vstack([X_train_base, X_train_hard])
    Y_train_raw = np.concatenate([Y_train_base, Y_train_hard])
    perm = rng.permutation(len(X_train_raw))
    X_train_raw, Y_train_raw = X_train_raw[perm], Y_train_raw[perm]

    train_df = pd.DataFrame(X_train_raw, columns=["tdb", "tr", "v", "rh"])
    train_df["pet"] = Y_train_raw

    test_df = pd.DataFrame(X_test_raw, columns=["tdb", "tr", "v", "rh"])
    test_df["pet"] = Y_test_raw

    train_csv = os.path.join(DATA_DIR, "train.csv")
    test_csv = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"saved {len(train_df)} training rows to {train_csv}")
    print(f"saved {len(test_df)} test rows to {test_csv}")

    save_distribution_plot(train_df, "Training set distributions",
                            os.path.join(DATA_DIR, "train_distributions.png"))
    save_distribution_plot(test_df, "Test set distributions",
                            os.path.join(DATA_DIR, "test_distributions.png"))


if __name__ == "__main__":
    main()
