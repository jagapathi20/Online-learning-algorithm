"""
generate_sample_data.py
───────────────────────
Synthetic streaming-data factory for the online-learning pipeline.

What it generates
─────────────────
A CSV with columns:

    feature_0, feature_1, ..., feature_{n-1}, label

where each row is one sample that will later be streamed one-at-a-time
by stream_loader.py.

Design knobs
────────────
    n_samples      – total rows in the CSV
    n_features     – width of the feature vector
    class_balance  – P(y=1).  0.5 = balanced; 0.3 = 30 % positive, etc.
    noise          – std of Gaussian noise added to the raw score before
                     thresholding.  Higher noise → harder problem.
    concept_drift  – if True, the *true* decision boundary flips sign at
                     the midpoint of the dataset.  This simulates the kind
                     of distribution shift an online model should adapt to
                     but a frozen batch model cannot.
    seed           – full reproducibility

How the labels are generated
────────────────────────────
1.  Draw a weight vector  w  ∈ Rⁿ  from N(0, 1).
2.  Draw features         X  from N(0, I).
3.  Compute raw scores    z  = X · w.
4.  Add noise             z  += N(0, noise²).
5.  Threshold             y  = 1  if  z > 0  else  0.
6.  (optional drift)      At row n_samples//2, negate w and repeat 3-5
    for the second half.  The positive-class fraction is then adjusted
    by flipping labels in the second half until class_balance is met.

Step 6 is intentionally simple: it gives a *detectable* drift that an
online learner with a non-zero learning rate will recover from, while a
batch model trained on the first half will degrade on the second half.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate(
    filepath: str | Path,
    n_samples: int = 5000,
    n_features: int = 5,
    class_balance: float = 0.5,
    noise: float = 1.0,
    concept_drift: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic streaming CSV and return it as a DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Where to write the CSV.  Parent directories are created if they
        do not exist.
    n_samples : int, default=5000
        Number of rows.
    n_features : int, default=5
        Number of feature columns.
    class_balance : float, default=0.5
        Target fraction of positive labels.  Clamped to (0.01, 0.99).
    noise : float, default=1.0
        Std of label noise.  0 = perfectly separable; higher = harder.
    concept_drift : bool, default=False
        If True, the true weight vector flips sign at the midpoint.
    seed : int, default=42
        Random seed.

    Returns
    -------
    df : pd.DataFrame
        The generated data (also written to *filepath*).

    Raises
    ------
    ValueError
        On invalid parameter combinations.
    """
    # ── validate ──────────────────────────────────────────────────────────
    if n_samples < 10:
        raise ValueError("n_samples must be ≥ 10.")
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1.")
    if not (0.0 < class_balance < 1.0):
        raise ValueError("class_balance must be in (0, 1).")
    if noise < 0:
        raise ValueError("noise must be ≥ 0.")

    class_balance = np.clip(class_balance, 0.01, 0.99)

    # ── RNG ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # ── true weight vector (the "signal") ─────────────────────────────────
    w = rng.standard_normal(n_features)
    # normalise so the signal magnitude is independent of n_features
    w = w / (np.linalg.norm(w) + 1e-10)

    # ── features: iid N(0,1) ──────────────────────────────────────────────
    X = rng.standard_normal((n_samples, n_features))

    # ── raw scores ────────────────────────────────────────────────────────
    z = X @ w + rng.normal(0, noise, size=n_samples)

    # ── labels from threshold ─────────────────────────────────────────────
    y = (z > 0).astype(np.int32)

    # ── concept drift: negate w at the midpoint ───────────────────────────
    mid = n_samples // 2

    if concept_drift:
        w_drifted = -w                                      # flipped boundary
        z_second  = X[mid:] @ w_drifted + rng.normal(0, noise, size=n_samples - mid)
        y[mid:]   = (z_second > 0).astype(np.int32)

    # ── enforce class_balance per half using independent sub-RNGs ─────────
    # Always split at mid so the first-half balance flips are identical
    # regardless of whether drift is on.  Each sub-RNG is keyed off the
    # main seed so the result is fully deterministic.
    bal_rng_1 = np.random.default_rng(seed + 1000)
    bal_rng_2 = np.random.default_rng(seed + 2000)
    y[:mid] = _enforce_balance(y[:mid], class_balance, bal_rng_1)
    y[mid:] = _enforce_balance(y[mid:], class_balance, bal_rng_2)

    # ── assemble DataFrame ────────────────────────────────────────────────
    col_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=col_names)
    df["label"] = y

    # ── write CSV ─────────────────────────────────────────────────────────
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    return df


def _enforce_balance(y: np.ndarray, target: float, rng) -> np.ndarray:
    """Flip the minimum number of labels to hit *target* positive rate.

    Strategy:
        * If too many positives → flip random 1→0 until balanced.
        * If too few  positives → flip random 0→1 until balanced.

    Parameters
    ----------
    y      : np.ndarray of int, shape (n,)
    target : float in (0, 1)
    rng    : numpy Generator

    Returns
    -------
    y : np.ndarray — same array, modified in place AND returned.
    """
    n        = len(y)
    n_target = int(round(target * n))
    n_pos    = int(y.sum())

    if n_pos == n_target:
        return y

    if n_pos > n_target:
        # too many 1s → flip some 1→0
        pos_indices = np.where(y == 1)[0]
        flip_count  = n_pos - n_target
        chosen      = rng.choice(pos_indices, size=flip_count, replace=False)
        y[chosen]   = 0
    else:
        # too few 1s → flip some 0→1
        neg_indices = np.where(y == 0)[0]
        flip_count  = n_target - n_pos
        chosen      = rng.choice(neg_indices, size=flip_count, replace=False)
        y[chosen]   = 1

    return y


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate the default sample CSVs used by the rest of the project
# ─────────────────────────────────────────────────────────────────────────────

def generate_defaults(data_dir: str | Path = "data/samples") -> dict[str, Path]:
    """Create the three standard CSVs the project uses out of the box.

    Returns a dict mapping short name → written path.

    Files
    -----
    basic.csv          – 5 000 rows, 5 features, balanced, moderate noise
    hard.csv           – 5 000 rows, 5 features, balanced, high noise
    drifted.csv        – 5 000 rows, 5 features, balanced, concept drift at midpoint
    """
    data_dir = Path(data_dir)
    paths = {}

    paths["basic"] = data_dir / "basic.csv"
    generate(
        filepath=paths["basic"],
        n_samples=5000, n_features=5,
        class_balance=0.5, noise=1.0,
        concept_drift=False, seed=42,
    )

    paths["hard"] = data_dir / "hard.csv"
    generate(
        filepath=paths["hard"],
        n_samples=5000, n_features=5,
        class_balance=0.5, noise=2.5,
        concept_drift=False, seed=42,
    )

    paths["drifted"] = data_dir / "drifted.csv"
    generate(
        filepath=paths["drifted"],
        n_samples=5000, n_features=5,
        class_balance=0.5, noise=1.0,
        concept_drift=True, seed=42,
    )

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    paths = generate_defaults()
    for name, path in paths.items():
        print(f"  wrote {name:>8} → {path}")
