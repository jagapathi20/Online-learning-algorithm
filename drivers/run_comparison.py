#!/usr/bin/env python3
"""
run_comparison.py
─────────────────
Driver script that runs both batch and online models on the same dataset,
plots the results, and optionally saves the comparison figures.

This is the primary entry point for evaluating how well the online learner
performs relative to a traditional batch-trained scikit-learn model.

Usage
─────
    python run_comparison.py \\
        --csv data/samples/basic.csv \\
        --window-size 500 \\
        --learning-rate 0.1 \\
        --lr-schedule invscale \\
        --decay 1e-3 \\
        --save-plots \\
        --output-dir plots/

What it does
────────────
1. Loads the specified CSV using batch_vs_online.compare()
2. Trains both models on the same data
3. Generates a multi-panel figure showing:
   - Accuracy over time (online sliding window vs batch test-set baseline)
   - Precision, Recall, F1 over time
   - AUC over time (when available)
   - Loss and learning rate curves
4. Prints a summary table comparing final metrics
5. Optionally saves the plots to disk

The plots give you a visual understanding of:
   - How quickly the online model converges
   - Whether it reaches batch-level performance
   - How metrics evolve over the stream (especially important for drift)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Path setup so we can import from evaluation/ and core/ ───────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # assumes run_comparison.py is in drivers/
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.batch_vs_online import compare


# ═════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════

def plot_comparison(results: Dict[str, Any], save_path: Path | None = None) -> None:
    """Create a comprehensive comparison plot.

    Parameters
    ----------
    results : dict
        The output of batch_vs_online.compare(), with keys:
            'batch'  → batch model results (test-set metrics)
            'online' → online model results (time-series histories)
            'csv_path' → path to the input CSV
    save_path : Path | None
        If provided, save the figure to this path.  Otherwise, display interactively.
    """
    batch = results["batch"]
    online = results["online"]
    histories = online["histories"]
    csv_name = Path(results["csv_path"]).name

    # ── figure setup ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Batch vs Online Logistic Regression — {csv_name}",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.35,
        wspace=0.25,
        top=0.94,
        bottom=0.06,
        left=0.06,
        right=0.98
    )

    # ── extract histories ─────────────────────────────────────────────────
    steps = np.array(histories["step"])
    acc = np.array(histories["accuracy"])
    prec = np.array(histories["precision"])
    rec = np.array(histories["recall"])
    f1 = np.array(histories["f1"])
    auc_vals = np.array(histories["auc"])
    loss = np.array(histories["loss"])
    lr_vals = np.array(histories["learning_rate"])

    # Handle None values in AUC (convert to NaN for plotting)
    auc_vals = np.array([x if x is not None else np.nan for x in auc_vals])

    # ── batch baselines (horizontal reference lines) ─────────────────────
    batch_acc = batch["accuracy"]
    batch_prec = batch["precision"]
    batch_rec = batch["recall"]
    batch_f1 = batch["f1"]
    batch_auc = batch["auc"]

    # ── PLOT 1: Accuracy ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, acc, label="Online (sliding window)", color="#2E86AB", linewidth=1.5)
    ax1.axhline(batch_acc, color="#A23B72", linestyle="--", linewidth=2, label="Batch (test set)")
    ax1.set_xlabel("Sample", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_title("Accuracy Over Time", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3, linestyle=":")
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 2: Precision ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, prec, label="Online (sliding window)", color="#2E86AB", linewidth=1.5)
    ax2.axhline(batch_prec, color="#A23B72", linestyle="--", linewidth=2, label="Batch (test set)")
    ax2.set_xlabel("Sample", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.set_title("Precision Over Time", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3, linestyle=":")
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 3: Recall ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, rec, label="Online (sliding window)", color="#2E86AB", linewidth=1.5)
    ax3.axhline(batch_rec, color="#A23B72", linestyle="--", linewidth=2, label="Batch (test set)")
    ax3.set_xlabel("Sample", fontsize=10)
    ax3.set_ylabel("Recall", fontsize=10)
    ax3.set_title("Recall Over Time", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(alpha=0.3, linestyle=":")
    ax3.set_ylim(0, 1.05)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 4: F1 Score ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(steps, f1, label="Online (sliding window)", color="#2E86AB", linewidth=1.5)
    ax4.axhline(batch_f1, color="#A23B72", linestyle="--", linewidth=2, label="Batch (test set)")
    ax4.set_xlabel("Sample", fontsize=10)
    ax4.set_ylabel("F1 Score", fontsize=10)
    ax4.set_title("F1 Score Over Time", fontsize=11, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(alpha=0.3, linestyle=":")
    ax4.set_ylim(0, 1.05)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 5: AUC ───────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    # Only plot non-NaN AUC values
    valid_auc = ~np.isnan(auc_vals)
    if valid_auc.any():
        ax5.plot(steps[valid_auc], auc_vals[valid_auc], 
                label="Online (sliding window)", color="#2E86AB", linewidth=1.5)
        ax5.axhline(batch_auc, color="#A23B72", linestyle="--", linewidth=2, label="Batch (test set)")
    else:
        ax5.text(0.5, 0.5, "AUC unavailable\n(need both classes in window)",
                ha="center", va="center", transform=ax5.transAxes, fontsize=10)
    ax5.set_xlabel("Sample", fontsize=10)
    ax5.set_ylabel("ROC-AUC", fontsize=10)
    ax5.set_title("ROC-AUC Over Time", fontsize=11, fontweight="bold")
    ax5.legend(loc="lower right", fontsize=9)
    ax5.grid(alpha=0.3, linestyle=":")
    ax5.set_ylim(0, 1.05)
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 6: Loss ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(steps, loss, color="#E63946", linewidth=1.5)
    ax6.set_xlabel("Sample", fontsize=10)
    ax6.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax6.set_title("Online Model Loss Over Time", fontsize=11, fontweight="bold")
    ax6.grid(alpha=0.3, linestyle=":")
    ax6.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 7: Learning Rate ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(steps, lr_vals, color="#F77F00", linewidth=1.5)
    ax7.set_xlabel("Sample", fontsize=10)
    ax7.set_ylabel("Learning Rate (η)", fontsize=10)
    ax7.set_title("Learning Rate Schedule", fontsize=11, fontweight="bold")
    ax7.grid(alpha=0.3, linestyle=":")
    ax7.set_yscale("log")  # Log scale often makes decay patterns clearer
    ax7.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    # ── PLOT 8: Summary Table ─────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis("off")

    # Get final online metrics (last recorded values)
    final_acc = acc[-1]
    final_prec = prec[-1]
    final_rec = rec[-1]
    final_f1 = f1[-1]
    final_auc = auc_vals[-1] if not np.isnan(auc_vals[-1]) else None

    # Build summary table
    table_data = [
        ["Metric", "Batch (Test Set)", "Online (Final Window)", "Δ"],
        ["Accuracy", f"{batch_acc:.4f}", f"{final_acc:.4f}", f"{final_acc - batch_acc:+.4f}"],
        ["Precision", f"{batch_prec:.4f}", f"{final_prec:.4f}", f"{final_prec - batch_prec:+.4f}"],
        ["Recall", f"{batch_rec:.4f}", f"{final_rec:.4f}", f"{final_rec - batch_rec:+.4f}"],
        ["F1 Score", f"{batch_f1:.4f}", f"{final_f1:.4f}", f"{final_f1 - batch_f1:+.4f}"],
    ]
    
    if final_auc is not None:
        table_data.append(
            ["ROC-AUC", f"{batch_auc:.4f}", f"{final_auc:.4f}", f"{final_auc - batch_auc:+.4f}"]
        )
    else:
        table_data.append(
            ["ROC-AUC", f"{batch_auc:.4f}", "N/A", "—"]
        )

    table = ax8.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    # Color-code delta column based on sign
    for i in range(1, len(table_data)):
        cell = table[(i, 3)]
        delta_text = table_data[i][3]
        if delta_text != "—" and delta_text != "N/A":
            delta_val = float(delta_text)
            if delta_val > 0:
                cell.set_facecolor("#D4EDDA")  # Light green
            elif delta_val < 0:
                cell.set_facecolor("#F8D7DA")  # Light red
            else:
                cell.set_facecolor("#FFF3CD")  # Light yellow

    ax8.set_title("Performance Summary", fontsize=11, fontweight="bold", pad=20)

    # ── Save or show ──────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print a text-based summary table to the console.

    Parameters
    ----------
    results : dict
        The output of batch_vs_online.compare()
    """
    batch = results["batch"]
    online = results["online"]
    histories = online["histories"]

    # Final online metrics
    final_acc = histories["accuracy"][-1]
    final_prec = histories["precision"][-1]
    final_rec = histories["recall"][-1]
    final_f1 = histories["f1"][-1]
    final_auc = histories["auc"][-1]

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Dataset: {results['csv_path']}")
    print(f"Total samples processed: {histories['step'][-1]}")
    print(f"Online window size: {online['evaluator'].window_size}")
    print(f"Online learning rate schedule: {online['model'].lr_schedule}")
    print(f"Online initial learning rate: {online['model'].lr0}")
    print("-" * 70)
    print(f"{'Metric':<15} {'Batch (Test)':<18} {'Online (Final)':<18} {'Delta':<10}")
    print("-" * 70)
    print(f"{'Accuracy':<15} {batch['accuracy']:<18.4f} {final_acc:<18.4f} {final_acc - batch['accuracy']:+.4f}")
    print(f"{'Precision':<15} {batch['precision']:<18.4f} {final_prec:<18.4f} {final_prec - batch['precision']:+.4f}")
    print(f"{'Recall':<15} {batch['recall']:<18.4f} {final_rec:<18.4f} {final_rec - batch['recall']:+.4f}")
    print(f"{'F1 Score':<15} {batch['f1']:<18.4f} {final_f1:<18.4f} {final_f1 - batch['f1']:+.4f}")
    
    if final_auc is not None:
        print(f"{'ROC-AUC':<15} {batch['auc']:<18.4f} {final_auc:<18.4f} {final_auc - batch['auc']:+.4f}")
    else:
        print(f"{'ROC-AUC':<15} {batch['auc']:<18.4f} {'N/A':<18} {'—':<10}")
    
    print("=" * 70)

    # Interpretation hints
    print("\nINTERPRETATION:")
    if final_acc >= batch['accuracy'] - 0.02:
        print("✓ Online model matches batch performance (within 2% on accuracy)")
    elif final_acc >= batch['accuracy'] - 0.05:
        print("⚠ Online model is close to batch performance (within 5% on accuracy)")
    else:
        print("✗ Online model underperforms batch (>5% gap on accuracy)")
        print("  → Consider tuning learning rate, increasing window size, or checking for concept drift")
    
    print()


# ═════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run batch vs online logistic regression comparison and plot results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (interactive plot)
  python run_comparison.py --csv data/samples/basic.csv

  # Save plots to file
  python run_comparison.py --csv data/samples/basic.csv --save-plots --output-dir plots/

  # Custom hyperparameters
  python run_comparison.py \\
      --csv data/samples/drifted.csv \\
      --window-size 500 \\
      --learning-rate 0.05 \\
      --lr-schedule adaptive \\
      --record-every 10

  # High-noise dataset with stronger regularization (via lower LR)
  python run_comparison.py --csv data/samples/hard.csv --learning-rate 0.01 --decay 1e-2
        """
    )

    # ── Required arguments ────────────────────────────────────────────────
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV file (generated by generate_sample_data.py)"
    )

    # ── Online model hyperparameters ──────────────────────────────────────
    parser.add_argument(
        "--window-size",
        type=int,
        default=500,
        help="Sliding window size for online metrics (default: 500)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Base learning rate η₀ for online SGD (default: 0.1)"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="invscale",
        choices=["constant", "invscale", "adaptive"],
        help="Learning rate schedule (default: invscale)"
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-3,
        help="Decay constant for invscale schedule (default: 1e-3)"
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record metrics every N steps (default: 1). Use higher values for large datasets."
    )

    # ── Batch model hyperparameters ───────────────────────────────────────
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for batch model testing (default: 0.2)"
    )

    # ── Output options ────────────────────────────────────────────────────
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to file instead of displaying interactively"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots/)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("BATCH VS ONLINE COMPARISON")
    print("=" * 70)
    print(f"Input CSV:         {csv_path}")
    print(f"Window size:       {args.window_size}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"LR schedule:       {args.lr_schedule}")
    print(f"Decay:             {args.decay}")
    print(f"Test size:         {args.test_size}")
    print(f"Record every:      {args.record_every} step(s)")
    print(f"Seed:              {args.seed}")
    print("=" * 70)

    # ── Run comparison ────────────────────────────────────────────────────
    print("\nRunning batch model (train/test split)...")
    print("Running online model (full sequential pass)...")
    
    results = compare(
        csv_path=csv_path,
        test_size=args.test_size,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        decay=args.decay,
        seed=args.seed,
        record_every=args.record_every,
    )

    print("✓ Comparison complete!")

    # ── Print summary table ───────────────────────────────────────────────
    print_summary_table(results)

    # ── Plot results ──────────────────────────────────────────────────────
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from CSV name and hyperparameters
        csv_stem = csv_path.stem
        filename = (
            f"{csv_stem}_w{args.window_size}_"
            f"lr{args.learning_rate}_{args.lr_schedule}.png"
        )
        save_path = output_dir / filename
        
        print(f"\nGenerating plots...")
        plot_comparison(results, save_path=save_path)
    else:
        print("\nGenerating plots (close window to exit)...")
        plot_comparison(results, save_path=None)

    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()
