"""
sliding_window_evaluator.py
───────────────────────────
Rolling-window classification metrics for online learning.

Why this exists
───────────────
In a streaming setting the model changes after every sample.  A single
cumulative accuracy number from t=0 onward has two problems:

    1.  Early mistakes get diluted — a wrong prediction at t=5 barely
        moves the needle at t=50 000.
    2.  Concept drift is invisible — if the underlying distribution
        shifts at t=10 000, cumulative accuracy keeps reporting the
        *average* over the old and new regimes.

A fixed-size sliding window solves both: it only scores the most
recent W predictions, so the metrics reflect *current* model quality.

Metrics computed
────────────────
    accuracy   = (TP + TN) / W
    precision  = TP / (TP + FP)          → 0 when no positive predictions
    recall     = TP / (TP + FN)          → 0 when no actual positives
    f1         = 2 · precision · recall / (precision + recall)   → 0 when both are 0
    auc        = area under the ROC curve (requires ≥ 2 distinct labels in window)

Implementation
──────────────
Uses collections.deque with maxlen=W as a circular buffer.  Appending
to a full deque automatically evicts the oldest entry — O(1) append,
O(W) metric computation.  No sorting, no reallocation.
"""

import numpy as np
from collections import deque
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Return type — a frozen snapshot of all metrics at one point in time
# ─────────────────────────────────────────────────────────────────────────────

class WindowMetrics(NamedTuple):
    """Immutable snapshot of metrics over the current window.

    Attributes
    ----------
    window_size : int       — number of predictions currently in the buffer (≤ W)
    accuracy    : float     — fraction correct in the window
    precision   : float     — TP / (TP + FP);  0.0 when no positive preds
    recall      : float     — TP / (TP + FN);  0.0 when no actual positives
    f1          : float     — harmonic mean of precision & recall;  0.0 when both 0
    auc         : float     — ROC-AUC;  None when < 2 distinct labels in window
    """
    window_size : int
    accuracy    : float
    precision   : float
    recall      : float
    f1          : float
    auc         : float | None


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowEvaluator:
    """Track classification metrics over a rolling window of predictions.

    Parameters
    ----------
    window_size : int, default=1000
        Number of most-recent predictions to keep.  Must be ≥ 2 (need at
        least 2 points to compute anything meaningful).
    """

    def __init__(self, window_size: int = 1000):
        if window_size < 2:
            raise ValueError("window_size must be ≥ 2.")

        self.window_size = window_size

        # Circular buffer: each entry is (predicted_label, true_label, predicted_prob)
        self._buffer: deque = deque(maxlen=window_size)

        # Cumulative counters — run alongside the window for a full-history view
        self._total_seen   : int = 0
        self._total_correct: int = 0

    # ── recording ─────────────────────────────────────────────────────────

    def record(self, predicted: int, actual: int, prob: float = 0.0) -> None:
        """Push one prediction into the window.

        Parameters
        ----------
        predicted : int   — model's predicted label (0 or 1)
        actual    : int   — ground-truth label (0 or 1)
        prob      : float — model's predicted P(y=1); used for AUC only
        """
        self._buffer.append((int(predicted), int(actual), float(prob)))
        self._total_seen += 1
        if predicted == actual:
            self._total_correct += 1

    # ── metrics ───────────────────────────────────────────────────────────

    def get_metrics(self) -> WindowMetrics:
        """Compute all metrics over the current window contents.

        Returns
        -------
        WindowMetrics  — a NamedTuple snapshot (see class docstring)
        """
        if len(self._buffer) == 0:
            return WindowMetrics(
                window_size=0,
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, auc=None
            )

        preds, actuals, probs = zip(*self._buffer)
        preds   = np.array(preds,   dtype=np.int32)
        actuals = np.array(actuals, dtype=np.int32)
        probs   = np.array(probs,   dtype=np.float64)

        # ── confusion-matrix counts ──
        tp = int(np.sum((preds == 1) & (actuals == 1)))
        tn = int(np.sum((preds == 0) & (actuals == 0)))
        fp = int(np.sum((preds == 1) & (actuals == 0)))
        fn = int(np.sum((preds == 0) & (actuals == 1)))

        n = len(self._buffer)

        accuracy  = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # ── AUC — only meaningful when both classes are present ──
        auc = self._compute_auc(actuals, probs)

        return WindowMetrics(
            window_size=n,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
        )

    # ── cumulative (full-history) accuracy ────────────────────────────────

    @property
    def cumulative_accuracy(self) -> float:
        """Accuracy over ALL predictions ever recorded (not just the window)."""
        if self._total_seen == 0:
            return 0.0
        return self._total_correct / self._total_seen

    # ── AUC helper ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_auc(actuals: np.ndarray, probs: np.ndarray) -> float | None:
        """ROC-AUC via the Wilcoxon-Mann-Whitney rank-sum method.

        Returns None when fewer than 2 distinct labels exist in the window
        — AUC is undefined in that case.

        Algorithm
        ---------
        Assign ranks to all samples by predicted probability (ascending).
        Ties receive the average of the ranks they would occupy.  AUC is
        then:

            AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2)
                  / (n_pos * n_neg)

        This is the standard tie-aware formulation and matches sklearn's
        roc_auc_score exactly.  O(n log n) due to the sort.
        """
        n_pos = int(np.sum(actuals == 1))
        n_neg = int(np.sum(actuals == 0))

        if n_pos == 0 or n_neg == 0:
            return None                         # AUC undefined

        n = len(actuals)

        # argsort ascending by probability
        order          = np.argsort(probs)
        sorted_probs   = probs[order]
        sorted_actuals = actuals[order]

        # assign ranks 1..n, averaging ties
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            # find the end of the current tie group
            while j < n and sorted_probs[j] == sorted_probs[i]:
                j += 1
            # average rank for every member of the tie group
            avg_rank = (i + 1 + j) / 2.0       # ranks are 1-based
            ranks[i:j] = avg_rank
            i = j

        # sum ranks where the true label is 1
        rank_sum_pos = float(np.sum(ranks[sorted_actuals == 1]))

        # Wilcoxon formula
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        return float(auc)

    # ── introspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Current number of entries in the window (≤ window_size)."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"SlidingWindowEvaluator(window_size={self.window_size}, "
            f"current={len(self._buffer)}, total_seen={self._total_seen})"
        )
