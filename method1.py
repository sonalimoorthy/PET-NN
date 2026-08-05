"""Method 1: PET surrogate with x/100-style input scaling.

Unlike the original notebook, scaling is NOT done outside the model: the
model itself takes raw physical inputs (tair, tmrt, rh, ws) and internally
computes whatever normalized features it needs, then internally unscales its
output (x100) so it returns PET directly in degC. No external
scale_X/scale_Y/unscale_Y anywhere in this script.

Run standalone (expects data/train.csv and data/test.csv, see
data_generation.py):
    python method1.py
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from data_generation import pet_reference

WS_MIN, WS_MAX = 0.0003, 10.0  # wind-speed range, needed for the internal log-scaling
SEED = 2025
DATA_DIR = "data"
OUTPUT_DIR = "output"
METHOD_NAME = "method1"


class MLP(nn.Module):
    width: int = 256

    @nn.compact
    def __call__(self, inputs):
        """inputs: raw [tair, tmrt, rh, ws], shape (batch, 4). Returns PET in degC."""
        init = nn.initializers.he_normal()
        act = nn.silu
        Tair, Tmrt, Rh, Ws = jnp.split(inputs, [1, 2, 3], axis=-1)  # 4 cols -> 4 pieces needs 3 cut points

        # -------- internal scaling (method 1: plain /100) --------
        Tair_s = Tair / 100.0
        Tmrt_s = Tmrt / 100.0
        Rh_s = Rh / 100.0
        Ws_l = Ws / 10.0
        Ws_log = (jnp.log(Ws + 1e-6) - jnp.log(WS_MIN)) / (jnp.log(WS_MAX) - jnp.log(WS_MIN))  # min-max normalized log-scale wind speed


        Temps = jnp.hstack([Tair_s, Tmrt_s])
        Temps = act(nn.Dense(128, kernel_init=init)(Temps))
        Temps = act(nn.Dense(self.width, kernel_init=init)(Temps))

        Air = jnp.hstack([Ws_l, Rh_s, Ws_log])
        Air = act(nn.Dense(self.width, kernel_init=init)(Air))
        Air = act(nn.Dense(self.width, kernel_init=init)(Air))

        merged = jnp.hstack([Temps, Air])
        merged = act(nn.Dense(self.width, kernel_init=init)(merged))
        merged = act(nn.Dense(self.width, kernel_init=init)(merged))

        output = nn.Dense(1, kernel_init=init)(merged)
        output = 100.0 * output  # unscale inside the model -- output is real degC
        return output


def make_batches(X, Y, idx, batch_size):
    n_batches = len(idx) // batch_size
    idx = idx[: n_batches * batch_size]
    Xb = X[idx].reshape(n_batches, batch_size, X.shape[-1])
    Yb = Y[idx].reshape(n_batches, batch_size, Y.shape[-1])
    return Xb, Yb


def train_model(model, optimizer, params_init, X_tr, Y_tr, X_val, Y_val,
                 epochs=1200, batch_size=512, eval_every=10, patience=None, seed=0, verbose=True):
    """X_tr/Y_tr/X_val/Y_val are raw (unscaled) units -- the model scales/unscales internally."""

    def loss_fn(params, x, y):
        preds = model.apply(params, x)
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def scan_epoch(params, opt_state, Xb, Yb):
        def step(carry, batch):
            params, opt_state = carry
            x, y = batch
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), (Xb, Yb))
        return params, opt_state, losses.mean()

    @jax.jit
    def predict(params, x):
        return model.apply(params, x)

    params = params_init
    opt_state = optimizer.init(params)
    X_tr_j, Y_tr_j = jnp.array(X_tr), jnp.array(Y_tr)
    X_val_j, Y_val_j = jnp.array(X_val), jnp.array(Y_val)

    rng = np.random.default_rng(seed)
    loss_history, mae_history = [], []
    best_mae, best_epoch, best_params = np.inf, 0, params
    t0 = time.perf_counter()

    for epoch in range(epochs):
        idx = rng.permutation(len(X_tr_j))
        Xb, Yb = make_batches(X_tr_j, Y_tr_j, idx, batch_size)
        params, opt_state, epoch_loss = scan_epoch(params, opt_state, Xb, Yb)
        loss_history.append(float(epoch_loss))

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_pred = np.array(predict(params, X_val_j))
            val_mae = float(np.mean(np.abs(val_pred - Y_val)))
            mae_history.append((epoch + 1, val_mae))
            if val_mae < best_mae:
                best_mae, best_epoch, best_params = val_mae, epoch + 1, params
            if verbose:
                print(f"  epoch {epoch + 1:5d}  loss {epoch_loss:.4f}  val_mae {val_mae:.4f}  "
                      f"({time.perf_counter() - t0:.1f}s)")
            if patience is not None and (epoch + 1) - best_epoch >= patience * eval_every:
                print(f"  early stop at epoch {epoch + 1} (best val MAE {best_mae:.4f} @ epoch {best_epoch})")
                break

    dt = time.perf_counter() - t0
    return best_params, loss_history, mae_history, best_mae, dt


def run_benchmark(model, params, test_df):
    """Accuracy metrics + NN-vs-pet_steady runtime speed-up, matching the
    notebooks' benchmark section."""
    X_test_raw = test_df[["tdb", "tr", "v", "rh"]].to_numpy(dtype=np.float64)
    pet_true = test_df["pet"].to_numpy(dtype=np.float64)

    # model expects columns in [tair, tmrt, rh, ws] order
    X_test_model = test_df[["tdb", "tr", "rh", "v"]].to_numpy(dtype=np.float64)
    pet_pred = np.array(model.apply(params, jnp.array(X_test_model))).reshape(-1)

    err = pet_pred - pet_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    max_ae = np.max(np.abs(err))
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(pet_true) * 100
    denom = np.maximum(np.abs(pet_true), 1.0)
    mape = np.mean(np.abs(err) / denom) * 100

    # ---- runtime: NN surrogate (batched, jitted) ----
    X_test_model_j = jnp.array(X_test_model)
    apply_jit = jax.jit(lambda p, x: model.apply(p, x))
    _ = apply_jit(params, X_test_model_j).block_until_ready()  # warm-up / compile

    n_rep = 20
    t0 = time.perf_counter()
    for _ in range(n_rep):
        _ = apply_jit(params, X_test_model_j).block_until_ready()
    t_nn = (time.perf_counter() - t0) / n_rep

    # ---- runtime: pythermalcomfort reference (sequential) ----
    t0 = time.perf_counter()
    for tdb, tr, v, rh in X_test_raw:
        pet_reference(tdb, tr, v, rh)
    t_ref = time.perf_counter() - t0

    summary = pd.DataFrame({
        "Metric": ["MAE (degC)", "RMSE (degC)", "Max abs error (degC)",
                   "Relative L2 error (%)", "Mean abs relative error (%)",
                   "NN runtime, 10k samples (s)", "pet_steady runtime, 10k samples (s)",
                   "Speed-up (x)"],
        "Value": [mae, rmse, max_ae, rel_l2, mape, t_nn, t_ref, t_ref / t_nn],
    })

    predictions_df = test_df[["tdb", "tr", "v", "rh"]].copy()
    predictions_df["pet_true"] = pet_true
    predictions_df["pet_pred"] = pet_pred
    predictions_df["abs_error"] = np.abs(err)

    return summary, predictions_df, pet_true, pet_pred


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # column order must match the model's jnp.split unpacking: [tair, tmrt, rh, ws]
    X_train = train_df[["tdb", "tr", "rh", "v"]].to_numpy(dtype=np.float64)
    Y_train = train_df[["pet"]].to_numpy(dtype=np.float64)
    X_test = test_df[["tdb", "tr", "rh", "v"]].to_numpy(dtype=np.float64)
    Y_test = test_df[["pet"]].to_numpy(dtype=np.float64)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=SEED)

    WIDTH = 256
    BATCH = 512
    EPOCHS = 1200

    model = MLP(width=WIDTH)
    key = jax.random.PRNGKey(0)
    params_init = model.init(key, jnp.ones((1, 4)))

    steps_per_epoch = len(X_tr) // BATCH
    max_iter = EPOCHS * steps_per_epoch
    sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=1e-3,
        warmup_steps=int(max_iter * 0.05), decay_steps=max_iter, end_value=1e-6,
    )
    optimizer = optax.adam(sched)

    print(f"=== {METHOD_NAME}: width={WIDTH} epochs={EPOCHS} batch={BATCH} "
          f"(raw-degC loss, scaling inside model) ===")
    params, loss_hist, mae_hist, best_mae, dt = train_model(
        model, optimizer, params_init, X_tr, Y_tr, X_val, Y_val,
        epochs=EPOCHS, batch_size=BATCH, patience=None,
    )
    print(f"\nTRAINING DONE in {dt:.1f}s, best val MAE = {best_mae:.4f}")

    val_pred = np.array(model.apply(params, jnp.array(X_val)))
    val_mae = float(np.mean(np.abs(val_pred - Y_val)))
    val_r2 = r2_score(Y_val, val_pred)

    test_pred = np.array(model.apply(params, jnp.array(X_test)))
    test_mae = float(np.mean(np.abs(test_pred - Y_test)))
    test_rmse = float(np.sqrt(np.mean((test_pred - Y_test) ** 2)))
    test_r2 = r2_score(Y_test, test_pred)

    print(f"VAL:  MAE={val_mae:.4f}  R2={val_r2:.6f}")
    print(f"TEST: MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  R2={test_r2:.6f}")

    model_path = os.path.join(OUTPUT_DIR, f"model_{METHOD_NAME}.pickle")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    print(f"saved {model_path}")

    print("\nRunning benchmark against pet_steady (this calls the real solver "
          "10,000 times sequentially, takes a while)...")
    summary, predictions_df, pet_true, pet_pred = run_benchmark(model, params, test_df)

    benchmark_path = os.path.join(OUTPUT_DIR, f"benchmark_{METHOD_NAME}.csv")
    summary.to_csv(benchmark_path, index=False)
    print(f"saved {benchmark_path}")
    print(summary.to_string(index=False))

    predictions_path = os.path.join(OUTPUT_DIR, f"predictions_{METHOD_NAME}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"saved {predictions_path}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pet_true, pet_pred, s=2, alpha=0.3)
    lims = [min(pet_true.min(), pet_pred.min()), max(pet_true.max(), pet_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("pet_steady PET (degC)")
    ax.set_ylabel("NN PET (degC)")
    ax.set_title(f"Parity plot (test set) - {METHOD_NAME}")
    parity_path = os.path.join(OUTPUT_DIR, f"parity_{METHOD_NAME}.png")
    plt.savefig(parity_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"saved {parity_path}")


if __name__ == "__main__":
    main()
