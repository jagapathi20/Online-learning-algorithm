"""
online_logistic_regression.py
─────────────────────────────
Online binary classifier using logistic regression updated via
Stochastic Gradient Descent (SGD), one sample at a time.

Design contract
───────────────
    1. predict(x)  →  (class_label, probability)   ← NO label seen yet
    2. update(x, y)                                 ← weights change AFTER prediction
    3. Learning-rate scheduling is pluggable (constant | invscale | adaptive)
    4. All state lives in plain NumPy arrays so the object is trivially
       serialisable and has zero hidden dependencies.

Math recap
──────────
    σ(z)   = 1 / (1 + exp(-z))                     # sigmoid
    ŷ      = σ(w · x)                              # predicted probability
    loss   = -[ y·log(ŷ) + (1-y)·log(1-ŷ) ]       # binary cross-entropy (per sample)
    ∂loss/∂w = (ŷ - y) · x                         # gradient for one sample
    w ← w - η · (ŷ - y) · x                       # SGD step
    b ← b - η · (ŷ - y)                           # bias update (same gradient trick)

Learning-rate schedules
───────────────────────
    constant   : η_t = η_0                          # simplest, may oscillate
    invscale   : η_t = η_0 / (1 + decay * t)       # classic decay, guaranteed convergence
    adaptive   : η_t = η_0 / sqrt(t)               # faster early learning, slower later
"""

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Numerical-stability helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip(z: np.ndarray, lo: float = -500.0, hi: float = 500.0) -> np.ndarray:
    """Clip raw logits so exp() never overflows or underflows to exactly 0."""
    return np.clip(z, lo, hi)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Uses the identity:
        σ(z) = 1 / (1 + exp(-z))            for z ≥ 0
        σ(z) = exp(z) / (1 + exp(z))        for z < 0   ← avoids exp(+large)
    """
    z = _clip(z)
    out = np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class OnlineLogisticRegression:
    """Logistic regression that learns one sample at a time via SGD.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.  Must be known upfront
        so the weight vector can be initialised.
    learning_rate : float, default=0.01
        Base learning rate η₀.
    lr_schedule : {'constant', 'invscale', 'adaptive'}, default='invscale'
        How η changes over time.  See module docstring for formulae.
    decay : float, default=1e-4
        Decay factor used only when lr_schedule='invscale'.
    threshold : float, default=0.5
        Probability cut-off for the positive class prediction.
    seed : int | None, default=42
        If not None, used to initialise weights with small random values
        instead of zeros.  Helps break symmetry in edge cases.

    Attributes
    -----------
    weights : np.ndarray of shape (n_features,)
        Current weight vector.
    bias : float
        Current bias (intercept) term.
    t : int
        Number of update steps performed so far (starts at 0).
    loss_history : list[float]
        Per-sample binary cross-entropy recorded at every update call.
    """

    VALID_SCHEDULES = ("constant", "invscale", "adaptive")

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        lr_schedule: str = "invscale",
        decay: float = 1e-4,
        threshold: float = 0.5,
        seed: int | None = 42,
    ):
        # ── validate ──────────────────────────────────────────────────────
        if n_features < 1:
            raise ValueError("n_features must be ≥ 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if lr_schedule not in self.VALID_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {self.VALID_SCHEDULES}, "
                f"got '{lr_schedule}'."
            )
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in (0, 1).")

        # ── hyper-parameters (frozen after init) ──────────────────────────
        self.n_features   = n_features
        self.lr0          = learning_rate
        self.lr_schedule  = lr_schedule
        self.decay        = decay
        self.threshold    = threshold

        # ── mutable state ─────────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        self.weights: np.ndarray = (
            rng.normal(0, 0.01, size=n_features) if seed is not None
            else np.zeros(n_features)
        )
        self.bias: float         = 0.0
        self.t: int              = 0            # global step counter
        self.loss_history: list  = []

    # ── learning-rate schedule ────────────────────────────────────────────

    @property
    def current_lr(self) -> float:
        """Return η for the *next* step (i.e. at step self.t)."""
        if self.lr_schedule == "constant":
            return self.lr0
        elif self.lr_schedule == "invscale":
            return self.lr0 / (1.0 + self.decay * self.t)
        else:                                   # adaptive  →  1/√t
            return self.lr0 / np.sqrt(1.0 + self.t)   # +1 avoids div-by-zero at t=0

    # ── core operations ───────────────────────────────────────────────────

    def _logit(self, x: np.ndarray) -> float:
        """Raw score z = w · x + b."""
        return float(np.dot(self.weights, x) + self.bias)

    def predict_proba(self, x: np.ndarray) -> float:
        """P(y = 1 | x) without touching any state."""
        return float(_sigmoid(np.array([self._logit(x)]))[0])

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """Return (class_label, probability).

        Parameters
        ----------
        x : array-like of shape (n_features,)

        Returns
        -------
        label : int   — 0 or 1
        prob  : float — P(y=1 | x)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._check_shape(x)

        prob  = self.predict_proba(x)
        label = int(prob >= self.threshold)
        return label, prob

    def update(self, x: np.ndarray, y: int) -> float:
        """Perform one SGD step and return the sample loss.

        Parameters
        ----------
        x : array-like of shape (n_features,)
        y : int — true label, 0 or 1

        Returns
        -------
        loss : float — binary cross-entropy for this sample (before the update)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = int(y)
        self._check_shape(x)

        # predicted probability (state unchanged)
        y_hat = self.predict_proba(x)

        # ── record loss BEFORE updating weights ──
        loss = self._binary_cross_entropy(y, y_hat)
        self.loss_history.append(loss)

        # ── SGD step ──
        eta   = self.current_lr
        error = y_hat - y                       # gradient factor

        self.weights -= eta * error * x         # weight update
        self.bias    -= eta * error             # bias update

        self.t += 1                             # advance step counter
        return loss

    # ── convenience: predict + update in one call ─────────────────────────

    def predict_and_update(self, x: np.ndarray, y: int) -> Tuple[int, float, float]:
        """predict() then update() — the standard online-learning loop step.

        Returns
        -------
        label : int
        prob  : float
        loss  : float
        """
        label, prob = self.predict(x)
        loss        = self.update(x, y)
        return label, prob, loss

    # ── static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _binary_cross_entropy(y: int, y_hat: float, eps: float = 1e-15) -> float:
        """Clip ŷ to (eps, 1-eps) so log() never hits 0."""
        y_hat = np.clip(y_hat, eps, 1.0 - eps)
        return float(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def _check_shape(self, x: np.ndarray) -> None:
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Expected feature vector of length {self.n_features}, "
                f"got {x.shape[0]}."
            )

    # ── state export / import ─────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a plain-dict snapshot of all mutable state."""
        return {
            "weights":       self.weights.tolist(),
            "bias":          self.bias,
            "t":             self.t,
            "loss_history":  list(self.loss_history),
        }

    def set_state(self, state: dict) -> None:
        """Restore mutable state from a dict previously returned by get_state()."""
        self.weights      = np.array(state["weights"], dtype=np.float64)
        self.bias         = float(state["bias"])
        self.t            = int(state["t"])
        self.loss_history = list(state["loss_history"])

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"OnlineLogisticRegression("
            f"n_features={self.n_features}, "
            f"lr0={self.lr0}, schedule='{self.lr_schedule}', "
            f"steps={self.t})"
        )
