"""
run_streaming.py
────────────────
Live streaming demo: the full online-learning loop running end-to-end
with a real-time terminal dashboard.

What happens when you run this
──────────────────────────────
    1.  A sample CSV is generated (if it does not already exist).
    2.  A StreamLoader reads it one row at a time.
    3.  Each row flows through:
            scale  →  predict  →  evaluate  →  update
    4.  Every *print_every* rows the terminal is cleared and a fresh
        metrics dashboard is printed.  This gives you a live view of
        the model learning in real time.

Usage
─────
    python -m drivers.run_streaming                          # defaults
    python -m drivers.run_streaming --csv data/samples/basic.csv
    python -m drivers.run_streaming --window 200 --print-every 50
    python -m drivers.run_streaming --lr 0.5 --schedule constant
    python -m drivers.run_streaming --help
"""

import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path

# ── make project root importable regardless of how this script is invoked ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.online_logistic_regression import OnlineLogisticRegression
from core.online_scaler              import OnlineScaler
from core.sliding_window_evaluator   import SlidingWindowEvaluator
from data.stream_loader              import StreamLoader
from data.generate_sample_data       import generate


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard renderer
# ─────────────────────────────────────────────────────────────────────────────

BAR_WIDTH = 40          # character width of the ASCII metric bars


def _bar(value: float, width: int = BAR_WIDTH) -> str:
    """Render a float in [0, 1] as a filled ASCII bar."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _render_dashboard(
    step: int,
    total: int,
    metrics,                    # WindowMetrics namedtuple
    loss: float,
    lr: float,
    cumulative_acc: float,
    elapsed: float,
) -> str:
    """Return the full dashboard string for one snapshot."""
    pct_done = 100.0 * step / total if total > 0 else 0.0

    auc_str = f"{metrics.auc:.3f}" if metrics.auc is not None else " N/A "

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║          ONLINE LOGISTIC REGRESSION — LIVE STREAM           ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  step {step:>6} / {total:<6}   ({pct_done:5.1f}%)   "
        f"elapsed {elapsed:6.2f}s                ║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  SLIDING-WINDOW METRICS                                     ║",
        f"║    accuracy   {_bar(metrics.accuracy)}  {metrics.accuracy:.3f}  ║",
        f"║    precision  {_bar(metrics.precision)}  {metrics.precision:.3f}  ║",
        f"║    recall     {_bar(metrics.recall)}  {metrics.recall:.3f}  ║",
        f"║    F1         {_bar(metrics.f1)}  {metrics.f1:.3f}  ║",
        f"║    AUC        {_bar(metrics.auc if metrics.auc else 0.0)}  {auc_str}  ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  cumulative accuracy  {cumulative_acc:.3f}                              ║",
        f"║  last sample loss     {loss:.4f}                             ║",
        f"║  current LR           {lr:.6f}                            ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming loop
# ─────────────────────────────────────────────────────────────────────────────

def run_stream(
    csv_path: str | Path,
    window_size: int = 500,
    learning_rate: float = 0.1,
    lr_schedule: str = "invscale",
    decay: float = 1e-3,
    print_every: int = 100,
    seed: int = 42,
) -> dict:
    """Execute the streaming loop and return a summary of what happened.

    Parameters
    ----------
    csv_path      – path to the CSV to stream
    window_size   – sliding window for metrics
    learning_rate – base η
    lr_schedule   – 'constant' | 'invscale' | 'adaptive'
    decay         – decay factor for invscale
    print_every   – how often (in rows) to refresh the dashboard
    seed          – weight-init seed for the online model

    Returns
    -------
    dict with keys: model, scaler, evaluator, final_metrics, total_rows, elapsed
    """
    csv_path = Path(csv_path)

    # ── count rows first so the dashboard can show progress ──────────────
    # (one cheap pass through the label column only)
    loader_count = StreamLoader(csv_path)
    total_rows   = loader_count.count_rows()

    # ── build components ──────────────────────────────────────────────────
    n_features = loader_count.n_features
    scaler     = OnlineScaler(n_features=n_features)
    model      = OnlineLogisticRegression(
                     n_features=n_features,
                     learning_rate=learning_rate,
                     lr_schedule=lr_schedule,
                     decay=decay,
                     seed=seed,
                 )
    evaluator  = SlidingWindowEvaluator(window_size=window_size)

    # StreamLoader with scaler wired in (handles fit_transform per row)
    loader = StreamLoader(csv_path, scaler=scaler)

    # ── streaming loop ────────────────────────────────────────────────────
    start_time = time.time()
    loss       = 0.0

    for step, (x, y) in enumerate(loader.stream(), start=1):
        # 1. predict  (state unchanged)
        pred, prob = model.predict(x)

        # 2. evaluate (record before weight update — prequential)
        evaluator.record(predicted=pred, actual=y, prob=prob)

        # 3. update weights
        loss = model.update(x, y)

        # 4. print dashboard at the requested cadence
        if step % print_every == 0 or step == total_rows:
            elapsed  = time.time() - start_time
            metrics  = evaluator.get_metrics()
            cum_acc  = evaluator.cumulative_accuracy
            dashboard = _render_dashboard(
                step, total_rows, metrics, loss, model.current_lr, cum_acc, elapsed
            )
            # clear terminal + print (works on Linux / macOS)
            print("\033[2J\033[H", end="")
            print(dashboard, flush=True)

    elapsed = time.time() - start_time

    return dict(
        model=model,
        scaler=scaler,
        evaluator=evaluator,
        final_metrics=evaluator.get_metrics(),
        total_rows=total_rows,
        elapsed=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live streaming demo for the online logistic regression pipeline."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file.  If omitted, generates data/samples/basic.csv automatically."
    )
    parser.add_argument(
        "--window", type=int, default=500,
        help="Sliding-window size for metrics (default: 500)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Base learning rate (default: 0.1)."
    )
    parser.add_argument(
        "--schedule", type=str, default="invscale",
        choices=["constant", "invscale", "adaptive"],
        help="Learning-rate schedule (default: invscale)."
    )
    parser.add_argument(
        "--decay", type=float, default=1e-3,
        help="Decay factor for invscale schedule (default: 0.001)."
    )
    parser.add_argument(
        "--print-every", type=int, default=100,
        help="Refresh the dashboard every N rows (default: 100)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for weight initialisation (default: 42)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── resolve CSV path — generate default if nothing was supplied ───────
    if args.csv is None:
        csv_path = PROJECT_ROOT / "data" / "samples" / "basic.csv"
        if not csv_path.exists():
            print(f"  generating {csv_path} …")
            generate(csv_path, n_samples=5000, n_features=5, noise=1.0, seed=42)
            print(f"  done.\n")
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"  ERROR: file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

    # ── run ───────────────────────────────────────────────────────────────
    result = run_stream(
        csv_path=csv_path,
        window_size=args.window,
        learning_rate=args.lr,
        lr_schedule=args.schedule,
        decay=args.decay,
        print_every=args.print_every,
        seed=args.seed,
    )

    # ── final summary (printed after the dashboard loop ends) ────────────
    m = result["final_metrics"]
    print(f"  ── DONE ──  {result['total_rows']} rows in {result['elapsed']:.2f}s")
    print(f"  final window metrics:  acc={m.accuracy:.3f}  f1={m.f1:.3f}  "
          f"auc={m.auc:.3f if m.auc else 'N/A'}")
    print(f"  cumulative accuracy:   {result['evaluator'].cumulative_accuracy:.3f}")
