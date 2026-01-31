# Online Logistic Regression — Project Structure

## Directory Layout

```
online_lr_project/
│
├── Project_structute.md        
│
├── core/                         ← pure algorithm logic (no I/O, no dependencies on other layers)
│   ├── __init__.py
│   ├── online_logistic_regression.py   ← [FILE 1] the SGD learner class
│   ├── online_scaler.py                ← [FILE 2] incremental feature standardisation (Welford)
│   └── sliding_window_evaluator.py    ← [FILE 3] rolling-window metrics tracker
│
├── data/                         ← data generation & loading utilities
│   ├── __init__.py
│   ├── generate_sample_data.py         ← [FILE 4] creates synthetic streaming CSVs
│   └── stream_loader.py                ← [FILE 5] row-by-row CSV reader (simulates a live stream)
│
├── evaluation/                   ← comparison & analysis layer
│   ├── __init__.py
│   └── batch_vs_online.py              ← [FILE 6] runs both models on same data, returns results
│
├── drivers/                      ← executable entry-points
│   ├── __init__.py
│   ├── run_streaming.py                ← [FILE 7] live streaming demo (predict → eval → update loop)
│   └── run_comparison.py              ← [FILE 8] batch-vs-online comparison runner + plot output
│
├── tests/                        ← unit tests for core layer
│   ├── __init__.py
│   ├── test_online_logistic_regression.py   ← [FILE 9]
│   ├── test_online_scaler.py                ← [FILE 10]
│   └── test_sliding_window_evaluator.py    ← [FILE 11]
│
├── requirements.txt              ← [FILE 12] pinned dependencies
└── README.md                     ← [FILE 13] usage & quick-start guide
```

---

## Build Order & Why

| Step | File(s)                          | Depends on          | Why this order                                                                   |
|------|----------------------------------|---------------------|----------------------------------------------------------------------------------|
| 1    | online_logistic_regression.py    | numpy only          | The heart of the project. Everything else feeds into or reads from it.           |
| 2    | online_scaler.py                 | numpy only          | Standalone math utility. Used by the stream loader before data hits the learner. |
| 3    | sliding_window_evaluator.py      | numpy, collections  | Standalone metrics buffer. Plugged into the streaming loop.                      |
| 4    | generate_sample_data.py          | numpy, pandas, sklearn | Produces the CSV the stream loader will read.                                 |
| 5    | stream_loader.py                 | pandas, core/scaler | Reads CSV row-by-row and optionally scales features on the fly.                  |
| 6    | batch_vs_online.py               | sklearn, core/*     | Orchestrates both training strategies on identical data.                         |
| 7    | run_streaming.py                 | core/*, data/*      | First runnable end-to-end demo.                                                  |
| 8    | run_comparison.py                | evaluation/*, matplotlib | Produces the visual comparison.                                             |
| 9-11 | test_*.py                        | pytest, core/*      | Written alongside their targets; run after each core file is done.               |
| 12   | requirements.txt                 | —                   | Finalized once all imports are known.                                            |
| 13   | README.md                        | —                   | Written last so it can reference actual usage.                                   |

---

## Data-Flow (single streaming iteration)

```
CSV row  ──►  stream_loader  ──►  online_scaler  ──►   ┌─────────────────────────┐
                                                       |  predict(x)  ──► ŷ      │
                                                       │       │                 │
                                                       │       ▼                 │
                                                       │  evaluator.record(ŷ, y) │   ← metrics updated BEFORE weights
                                                       │       │                 │
                                                       │       ▼                 │
                                                       │  update(x, y)  ──► w'   │   ← weights change AFTER eval
                                                       └─────────────────────────┘
```

This **predict → evaluate → update** ordering is intentional:
it gives you *prequential* (honest, no-look-ahead) evaluation — every
prediction is scored against ground truth *before* the model sees that label.
