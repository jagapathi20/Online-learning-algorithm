"""
stream_loader.py
────────────────
Row-by-row CSV reader that simulates a live data stream.

Why this exists
───────────────
pandas.read_csv() loads an entire file into memory at once.  In a real
streaming system the data arrives one (or a few) rows at a time and
never all at once.  This module wraps pandas' chunked reader to honour
that contract: it yields one sample per iteration, keeping memory usage
constant regardless of file size.

Pipeline position
─────────────────
    CSV on disk  →  StreamLoader  →  (features, label)  →  online loop
                        │
                        ├── optional OnlineScaler.fit_transform() per row
                        │   (updates running stats AND scales in one step)
                        └── yields numpy arrays, not DataFrames

Design decisions
────────────────
* chunk_size defaults to 1 (true single-row streaming).  Set higher for
  micro-batches if I/O latency matters.
* The scaler is optional.  Pass scaler=None to get raw features.
* feature_columns / label_column are auto-detected from the CSV header
  if not supplied explicitly.  The convention is that every column whose
  name starts with "feature_" is a feature and "label" is the target.
* The generator is re-entrant: call stream() multiple times on the same
  loader and it restarts from the top of the file each time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator

# relative import so this module can live inside the data/ package
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.online_scaler import OnlineScaler


class StreamLoader:
    """Iterate over a CSV file one row at a time, optionally scaling on the fly.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.  Must exist.
    label_column : str, default='label'
        Name of the target column.
    feature_columns : list[str] | None, default=None
        Which columns are features.  If None, every column that is NOT
        the label column is treated as a feature (order preserved as in
        the CSV).
    chunk_size : int, default=1
        Number of rows to read from disk per I/O call.  Each row is still
        yielded individually — this only controls the internal read buffer.
    scaler : OnlineScaler | None, default=None
        If provided, each feature vector is passed through
        scaler.fit_transform() before being yielded.  The scaler's
        running statistics are updated as a side-effect.
    """

    def __init__(
        self,
        filepath: str | Path,
        label_column: str = "label",
        feature_columns: list[str] | None = None,
        chunk_size: int = 1,
        scaler: OnlineScaler | None = None,
    ):
        self.filepath       = Path(filepath)
        self.label_column   = label_column
        self.chunk_size     = chunk_size
        self.scaler         = scaler

        # ── validate file exists ──────────────────────────────────────────
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV not found: {self.filepath}")

        # ── peek at the header to resolve feature columns ────────────────
        header = pd.read_csv(self.filepath, nrows=0).columns.tolist()

        if label_column not in header:
            raise ValueError(
                f"Label column '{label_column}' not found in {self.filepath}. "
                f"Available columns: {header}"
            )

        if feature_columns is not None:
            missing = set(feature_columns) - set(header)
            if missing:
                raise ValueError(
                    f"Feature columns {missing} not found in {self.filepath}."
                )
            self.feature_columns = list(feature_columns)
        else:
            # auto-detect: everything that isn't the label
            self.feature_columns = [c for c in header if c != label_column]

        if len(self.feature_columns) == 0:
            raise ValueError("No feature columns detected — file has only the label column.")

        # ── validate scaler dimensions if one was provided ───────────────
        if self.scaler is not None:
            if self.scaler.n_features != len(self.feature_columns):
                raise ValueError(
                    f"Scaler expects {self.scaler.n_features} features but "
                    f"CSV has {len(self.feature_columns)}: {self.feature_columns}"
                )

        # ── cached metadata ───────────────────────────────────────────────
        self.n_features = len(self.feature_columns)

    # ── main generator ────────────────────────────────────────────────────

    def stream(self) -> Generator[tuple[np.ndarray, int], None, None]:
        """Yield (feature_vector, label) one row at a time.

        The generator re-reads the file from the top every time it is
        called, so you can iterate multiple times over the same loader.

        Yields
        ------
        x : np.ndarray of shape (n_features,)
            Feature vector (scaled if a scaler was provided).
        y : int
            Binary label (0 or 1).
        """
        reader = pd.read_csv(
            self.filepath,
            chunksize=self.chunk_size,
            usecols=self.feature_columns + [self.label_column],
        )

        for chunk in reader:
            for _, row in chunk.iterrows():
                x = row[self.feature_columns].values.astype(np.float64)
                y = int(row[self.label_column])

                # optional online scaling
                if self.scaler is not None:
                    x = self.scaler.fit_transform(x)

                yield x, y

    # ── convenience: count rows without loading everything ───────────────

    def count_rows(self) -> int:
        """Return total number of data rows (excludes header).

        Uses chunked reading so memory stays constant even for huge files.
        """
        total = 0
        for chunk in pd.read_csv(self.filepath, chunksize=10_000, usecols=[self.label_column]):
            total += len(chunk)
        return total

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        scaled = "scaled" if self.scaler else "raw"
        return (
            f"StreamLoader(file='{self.filepath.name}', "
            f"features={self.n_features}, chunk_size={self.chunk_size}, {scaled})"
        )
