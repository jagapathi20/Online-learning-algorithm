"""
online_scaler.py
────────────────
Incremental feature standardisation for streaming data.

Why this exists
───────────────
SGD is sensitive to feature magnitudes.  A traditional StandardScaler
requires a full pass over the data to compute mean and variance — that
is not possible in a streaming setting.  This class maintains *running*
statistics and updates them in O(1) time and O(features) space per
sample, using Welford's online algorithm.

Welford's algorithm (per feature j)
────────────────────────────────────
    n       ← n + 1
    δ       ← x_j - mean_j
    mean_j  ← mean_j + δ / n
    δ'      ← x_j - mean_j          # note: uses the NEW mean
    M2_j    ← M2_j + δ * δ'

    variance_j = M2_j / n           # population variance
    std_j      = sqrt(variance_j)

Transform
─────────
    x_scaled_j = (x_j - mean_j) / (std_j + ε)

    ε (epsilon) prevents division by zero when a feature is constant
    across all samples seen so far.

Design notes
────────────
* fit_transform(x) and transform(x) are intentionally separate so the
  streaming driver can call fit_transform (update stats THEN scale) on
  every incoming sample, while batch code can call fit on a block and
  then transform without changing stats.
* partial_fit(x) is an alias for the stats-update step alone, matching
  the scikit-learn partial_fit convention.
* All state is plain NumPy arrays → trivially serialisable.
"""

import numpy as np


class OnlineScaler:
    """Standardise features using Welford's incremental algorithm.

    Parameters
    ----------
    n_features : int
        Number of features.  Must match the width of every vector passed
        to fit / transform.
    epsilon : float, default=1e-8
        Added to std in the denominator to avoid division by zero for
        constant features.

    Attributes
    ----------
    n_samples_seen : int
        Total number of samples that have been passed to fit/partial_fit.
    mean_ : np.ndarray of shape (n_features,)
        Running mean per feature.
    M2_ : np.ndarray of shape (n_features,)
        Running sum of squared deviations (M2 in Welford's notation).
    var_ : np.ndarray of shape (n_features,)
        Current population variance per feature (read-only property).
    std_ : np.ndarray of shape (n_features,)
        Current population std per feature (read-only property).
    """

    def __init__(self, n_features: int, epsilon: float = 1e-8):
        if n_features < 1:
            raise ValueError("n_features must be ≥ 1.")

        self.n_features     = n_features
        self.epsilon        = epsilon

        # mutable state — all zeros at start
        self.n_samples_seen: int        = 0
        self.mean_: np.ndarray          = np.zeros(n_features, dtype=np.float64)
        self.M2_  : np.ndarray          = np.zeros(n_features, dtype=np.float64)

    # ── read-only derived statistics ──────────────────────────────────────

    @property
    def var_(self) -> np.ndarray:
        """Population variance per feature.  Zero before any sample is seen."""
        if self.n_samples_seen == 0:
            return np.zeros(self.n_features)
        return self.M2_ / self.n_samples_seen

    @property
    def std_(self) -> np.ndarray:
        """Population std per feature."""
        return np.sqrt(self.var_)

    # ── core: Welford update ──────────────────────────────────────────────

    def partial_fit(self, x: np.ndarray) -> "OnlineScaler":
        """Absorb one sample into the running statistics (no transform).

        This is the raw Welford step.  Call this when you only want to
        update stats without immediately scaling the vector.

        Parameters
        ----------
        x : array-like of shape (n_features,)

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._check_shape(x)

        self.n_samples_seen += 1
        delta      = x - self.mean_                         # δ  = x − old_mean
        self.mean_ += delta / self.n_samples_seen           # mean ← mean + δ/n
        delta2     = x - self.mean_                         # δ' = x − new_mean
        self.M2_  += delta * delta2                         # M2 ← M2 + δ·δ'

        return self

    # ── transform (no state change) ───────────────────────────────────────

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale x using the *current* running statistics.

        Safe to call before any fit — returns x unchanged (mean=0, std=0
        triggers the epsilon guard, so division produces x/ε which is
        numerically large; to avoid that, we special-case n_samples_seen < 2
        and return x unscaled).

        Parameters
        ----------
        x : array-like of shape (n_features,)

        Returns
        -------
        x_scaled : np.ndarray of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._check_shape(x)

        if self.n_samples_seen < 2:
            # not enough data to estimate variance; return raw
            return x.copy()

        return (x - self.mean_) / (self.std_ + self.epsilon)

    # ── combined: update stats THEN scale ─────────────────────────────────

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """partial_fit(x) then transform(x) — the standard streaming step.

        The sample is absorbed into the running stats *first*, so the
        scaling it receives already reflects its own contribution.  This
        matches the "see everything up to and including now" semantics
        expected in an online pipeline.

        Parameters
        ----------
        x : array-like of shape (n_features,)

        Returns
        -------
        x_scaled : np.ndarray of shape (n_features,)
        """
        self.partial_fit(x)
        return self.transform(x)

    # ── state export / import ─────────────────────────────────────────────

    def get_state(self) -> dict:
        """Plain-dict snapshot of all mutable state."""
        return {
            "n_samples_seen": self.n_samples_seen,
            "mean_":          self.mean_.tolist(),
            "M2_":            self.M2_.tolist(),
        }

    def set_state(self, state: dict) -> None:
        """Restore from a dict previously returned by get_state()."""
        self.n_samples_seen = int(state["n_samples_seen"])
        self.mean_          = np.array(state["mean_"], dtype=np.float64)
        self.M2_            = np.array(state["M2_"],   dtype=np.float64)

    # ── helpers ───────────────────────────────────────────────────────────

    def _check_shape(self, x: np.ndarray) -> None:
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Expected vector of length {self.n_features}, got {x.shape[0]}."
            )

    def __repr__(self) -> str:
        return (
            f"OnlineScaler(n_features={self.n_features}, "
            f"n_samples_seen={self.n_samples_seen})"
        )
