"""
batch_vs_online.py
──────────────────
Orchestrator that runs both a sklearn batch model and our online learner
on identical data and returns comparable results.

Why two separate runs on the same data?
────────────────────────────────────────
    Batch model  – sees ALL training rows at once, fits in one go, then
                   is evaluated once on a held-out test set.  This is the
                   gold-standard accuracy ceiling for a linear model on
                   this feature set.

    Online model – sees each row exactly once, in order.  It predicts
                   BEFORE seeing the label (prequential / honest eval),
                   then updates.  We record sliding-window metrics at
                   every step so we can plot the convergence curve.

What this module returns
────────────────────────
A single dict with two top-level keys:

    results["batch"]  – dict with the sklearn model's test-set metrics
    results["online"] – dict with time-series histories (one value per
                        sample) so the driver can plot accuracy, F1, AUC,
                        loss, and the learning rate over time.

Importantly this module does NO plotting — it is pure logic.  The
driver (run_comparison.py) owns the matplotlib calls.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.online_logistic_regression  import OnlineLogisticRegression
from core.online_scaler               import OnlineScaler
from core.sliding_window_evaluator    import SlidingWindowEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train sklearn LogisticRegression on a train split, evaluate on test.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    test_size : float   – fraction held out for evaluation
    seed : int          – controls the train/test split

    Returns
    -------
    dict with keys:
        model          – the fitted sklearn model
        scaler         – the fitted sklearn StandardScaler
        X_train, y_train, X_test, y_test  – the split arrays
        accuracy, precision, recall, f1, auc  – test-set metrics (float)
    """
    # ── split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ── scale (fit on train only — no data leakage) ──────────────────────
    scaler   = SklearnScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── fit ───────────────────────────────────────────────────────────────
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train_s, y_train)

    # ── evaluate on test set ──────────────────────────────────────────────
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    return dict(
        model=model,
        scaler=scaler,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        accuracy  = float(accuracy_score(y_test, y_pred)),
        precision = float(precision_score(y_test, y_pred, zero_division=0)),
        recall    = float(recall_score(y_test, y_pred, zero_division=0)),
        f1        = float(f1_score(y_test, y_pred, zero_division=0)),
        auc       = float(roc_auc_score(y_test, y_proba)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Online runner
# ─────────────────────────────────────────────────────────────────────────────

def run_online(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 500,
    learning_rate: float = 0.1,
    lr_schedule: str = "invscale",
    decay: float = 1e-3,
    seed: int = 42,
    record_every: int = 1,
) -> dict:
    """Run the full prequential online loop: predict → eval → update.

    Parameters
    ----------
    X, y           – full dataset (order matters — this IS the stream)
    window_size    – sliding-window size for the evaluator
    learning_rate  – base η for the online learner
    lr_schedule    – one of 'constant', 'invscale', 'adaptive'
    decay          – decay factor (used when lr_schedule='invscale')
    seed           – passed to the online learner for weight init
    record_every   – record metrics every N steps (1 = every step).
                     Use higher values on large datasets to keep histories
                     manageable.

    Returns
    -------
    dict with keys:
        model, scaler, evaluator  – final state of each component
        histories – dict of lists, one entry per recorded step:
            step, accuracy, precision, recall, f1, auc, loss, learning_rate
    """
    n_samples, n_features = X.shape

    # ── components ────────────────────────────────────────────────────────
    scaler    = OnlineScaler(n_features=n_features)
    model     = OnlineLogisticRegression(
                    n_features=n_features,
                    learning_rate=learning_rate,
                    lr_schedule=lr_schedule,
                    decay=decay,
                    seed=seed,
                )
    evaluator = SlidingWindowEvaluator(window_size=window_size)

    # ── history accumulators ──────────────────────────────────────────────
    hist = {
        "step":          [],
        "accuracy":      [],
        "precision":     [],
        "recall":        [],
        "f1":            [],
        "auc":           [],
        "loss":          [],
        "learning_rate": [],
    }

    # ── prequential loop ──────────────────────────────────────────────────
    for t in range(n_samples):
        x_raw = X[t]
        label = int(y[t])

        # 1. scale (updates running stats, then transforms)
        x_scaled = scaler.fit_transform(x_raw)

        # 2. predict (no label seen yet)
        pred, prob = model.predict(x_scaled)

        # 3. record into sliding window (before weight update)
        evaluator.record(predicted=pred, actual=label, prob=prob)

        # 4. update weights
        loss = model.update(x_scaled, label)

        # 5. snapshot metrics at the requested cadence
        if (t + 1) % record_every == 0:
            m = evaluator.get_metrics()
            hist["step"].append(t + 1)
            hist["accuracy"].append(m.accuracy)
            hist["precision"].append(m.precision)
            hist["recall"].append(m.recall)
            hist["f1"].append(m.f1)
            hist["auc"].append(m.auc)          # may be None early on
            hist["loss"].append(loss)
            hist["learning_rate"].append(model.current_lr)

    return dict(
        model=model,
        scaler=scaler,
        evaluator=evaluator,
        histories=hist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level: run both and package into one results dict
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    csv_path: str | Path,
    test_size: float = 0.2,
    window_size: int = 500,
    learning_rate: float = 0.1,
    lr_schedule: str = "invscale",
    decay: float = 1e-3,
    seed: int = 42,
    record_every: int = 1,
) -> dict:
    """Load a CSV, run batch and online on it, return unified results.

    Parameters
    ----------
    csv_path       – path to a CSV generated by generate_sample_data
    test_size      – fraction for the batch model's held-out test set
    window_size    – sliding window for online metrics
    learning_rate  – base η for online SGD
    lr_schedule    – LR schedule name
    decay          – decay constant
    seed           – shared seed for reproducibility
    record_every   – how often to snapshot online metrics

    Returns
    -------
    dict with keys:
        "batch"  – output of run_batch()
        "online" – output of run_online()
        "csv_path" – the input path (for labelling plots later)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ── load once, share between both runners ─────────────────────────────
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int32)

    # ── run both ──────────────────────────────────────────────────────────
    batch_results  = run_batch(X, y, test_size=test_size, seed=seed)
    online_results = run_online(
        X, y,
        window_size=window_size,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        decay=decay,
        seed=seed,
        record_every=record_every,
    )

    return dict(
        batch=batch_results,
        online=online_results,
        csv_path=str(csv_path),
    )
