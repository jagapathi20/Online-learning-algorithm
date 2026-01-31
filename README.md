# Online Logistic Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready implementation of **online (streaming) logistic regression** for binary classification. Unlike traditional batch learning, this system processes data samples one at a time with constant memory usage, making it ideal for real-time prediction systems, large datasets, and non-stationary environments.

## Key Features

- **Stochastic Gradient Descent (SGD)** with three learning rate schedules (constant, invscale, adaptive)
- **Incremental feature standardization** using Welford's online algorithm
- **Sliding-window evaluation** for tracking model performance over time
- **Prequential evaluation** ensuring honest metrics (no look-ahead bias)
- **Direct comparison framework** with scikit-learn batch models
- **Comprehensive visualization** of convergence and performance metrics
- **Zero external dependencies** for core algorithms (NumPy only)

---

##  Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Core Components](#core-components)
- [Understanding the Results](#understanding-the-results)
- [Advanced Usage](#advanced-usage)
- [Performance Characteristics](#performance-characteristics)
- [When to Use Online Learning](#when-to-use-online-learning)
- [Contributing](#contributing)
- [License](#license)

---

##  Installation

### Requirements

- Python 3.10 or higher
- pip

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/online-learning-algorithm.git
cd online-learning-algorithm

# Install required packages
pip install -r requirements.txt
```

### Development Installation 

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### 1. Generate Sample Data

```bash
python data/generate_sample_data.py
```

This creates three sample datasets in `data/samples/`:
- `basic.csv` — 5,000 samples, moderate noise, stationary distribution
- `hard.csv` — 5,000 samples, high noise, challenging for linear models
- `drifted.csv` — 5,000 samples, concept drift at t=2,500

### 2. Run Batch vs. Online Comparison

```bash
python drivers/run_comparison.py --csv data/samples/basic.csv
```

This will:
1. Train a scikit-learn batch model (train/test split)
2. Train the online model (sequential processing)
3. Display an 8-panel comparison plot showing:
   - Accuracy, Precision, Recall, F1, AUC over time
   - Loss and learning rate curves
   - Summary table comparing final metrics

### 3. Save Comparison Plots

```bash
python drivers/run_comparison.py \
    --csv data/samples/basic.csv \
    --save-plots \
    --output-dir plots/
```

---

##  Project Structure

```
online_lr_project/
│
├── core/                         # Pure algorithm implementations
│   ├── online_logistic_regression.py   # SGD learner
│   ├── online_scaler.py                # Welford feature standardization
│   └── sliding_window_evaluator.py    # Rolling metrics tracker
│
├── data/                         # Data generation & streaming
│   ├── generate_sample_data.py         # Synthetic data factory
│   └── stream_loader.py                # Row-by-row CSV reader
│
├── evaluation/                   # Comparison framework
│   └── batch_vs_online.py              # Runs both models on same data
│
├── drivers/                      # Executable entry points
│   ├── run_streaming.py                # Live streaming demo
│   └── run_comparison.py              # Comparison runner + plots
│
│
├── requirements.txt              # Pinned dependencies
└── README.md                     # This file
```

---

##  Usage Examples

### Example 1: Basic Comparison

```bash
python drivers/run_comparison.py --csv data/samples/basic.csv
```

**Output:**
- Interactive plot with 8 panels
- Console summary table
- Performance interpretation hints

### Example 2: Concept Drift Detection

```bash
python drivers/run_comparison.py \
    --csv data/samples/drifted.csv \
    --window-size 300 \
    --learning-rate 0.1 \
    --lr-schedule constant
```

**Use case:** The constant learning rate maintains plasticity, allowing the model to adapt when the distribution shifts at t=2,500.

### Example 3: High-Noise Dataset

```bash
python drivers/run_comparison.py \
    --csv data/samples/hard.csv \
    --learning-rate 0.01 \
    --decay 1e-2
```

**Use case:** Lower learning rate and higher decay improve stability on noisy data.

### Example 4: Programmatic Usage

```python
from core.online_logistic_regression import OnlineLogisticRegression
from core.online_scaler import OnlineScaler
from data.stream_loader import StreamLoader

# Initialize components
scaler = OnlineScaler(n_features=5)
model = OnlineLogisticRegression(
    n_features=5,
    learning_rate=0.1,
    lr_schedule='invscale'
)
loader = StreamLoader('data.csv', scaler=scaler)

# Streaming loop
for x, y in loader.stream():
    # Predict (without seeing the label)
    pred, prob = model.predict(x)
    
    # Update weights (SGD step)
    loss = model.update(x, y)
    
    print(f"Predicted: {pred}, Actual: {y}, Loss: {loss:.4f}")
```

### Example 5: Custom Data Generation

```python
from data.generate_sample_data import generate

generate(
    filepath='my_data.csv',
    n_samples=10000,
    n_features=10,
    class_balance=0.3,  # 30% positive class
    noise=1.5,
    concept_drift=True,
    seed=123
)
```

---

##  Core Components

### OnlineLogisticRegression

**File:** `core/online_logistic_regression.py`

Implements binary classification via SGD with:
- Numerically stable sigmoid computation
- Three learning rate schedules:
  - `constant`: η_t = η₀
  - `invscale`: η_t = η₀ / (1 + decay·t)
  - `adaptive`: η_t = η₀ / √t
- Configurable classification threshold
- State serialization (get_state/set_state)

**Mathematical Foundation:**

```
σ(z) = 1 / (1 + exp(-z))                    [sigmoid]
ŷ = σ(w·x + b)                              [prediction]
loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]         [binary cross-entropy]
∂loss/∂w = (ŷ - y)·x                        [gradient]
w ← w - η·(ŷ - y)·x                         [SGD update]
```

### OnlineScaler

**File:** `core/online_scaler.py`

Incremental feature standardization using Welford's algorithm:
- Computes running mean and variance in O(1) time per sample
- No need for multiple passes over data
- Numerically stable for large datasets
- Memory: O(n_features) regardless of stream length

**Welford's Algorithm:**

```
n ← n + 1
δ ← x_j - mean_j
mean_j ← mean_j + δ/n
δ' ← x_j - mean_j         [new mean]
M2_j ← M2_j + δ·δ'
var_j = M2_j / n

x_scaled_j = (x_j - mean_j) / (std_j + ε)
```

### SlidingWindowEvaluator

**File:** `core/sliding_window_evaluator.py`

Tracks classification metrics over a rolling window:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Uses `collections.deque` for O(1) append
- Handles tied probabilities in AUC computation
- Returns `None` for AUC when < 2 distinct labels in window

---

##  Understanding the Results

### Comparison Plot Panels

1. **Accuracy Over Time**: Shows how the online model converges to batch performance
2. **Precision/Recall/F1**: Tracks class-specific performance metrics
3. **ROC-AUC**: Probability calibration quality (when both classes present)
4. **Loss Curve**: Binary cross-entropy over time (should decrease)
5. **Learning Rate Schedule**: Visualizes η decay pattern
6. **Summary Table**: Final metric comparison with color-coded deltas

### Interpreting the Summary Table

```
Metric      Batch (Test)    Online (Final)    Δ
─────────────────────────────────────────────────
Accuracy    0.8750          0.8720            -0.0030
Precision   0.8800          0.8750            -0.0050
Recall      0.8700          0.8690            -0.0010
F1 Score    0.8750          0.8720            -0.0030
ROC-AUC     0.9300          0.9280            -0.0020
```

**Green delta** = Online outperforms batch (rare on stationary data)  
**Red delta** = Batch outperforms online (common, gap should be small)  
**Yellow delta** = Exactly equal

**Good performance:** Online within 2-5% of batch on stationary data  
**Excellent performance:** Online matches or exceeds batch (suggests drift adaptation)

---

##  Advanced Usage

### Learning Rate Selection

```bash
# Fast convergence on stationary data
python drivers/run_comparison.py \
    --csv data/samples/basic.csv \
    --lr-schedule invscale \
    --learning-rate 0.1 \
    --decay 1e-2

# Adaptation to concept drift
python drivers/run_comparison.py \
    --csv data/samples/drifted.csv \
    --lr-schedule constant \
    --learning-rate 0.05

# Aggressive early learning
python drivers/run_comparison.py \
    --csv data/samples/basic.csv \
    --lr-schedule adaptive \
    --learning-rate 0.2
```

### Window Size Tuning

- **Small window (100-300)**: Faster drift detection, noisier metrics
- **Medium window (500-1000)**: Balanced stability and responsiveness
- **Large window (2000+)**: Smoother metrics, slower drift response

```bash
# Drift-sensitive configuration
python drivers/run_comparison.py \
    --csv data/samples/drifted.csv \
    --window-size 200

# Stable metrics on stationary data
python drivers/run_comparison.py \
    --csv data/samples/basic.csv \
    --window-size 1000
```

### Model Persistence

```python
import json
from core.online_logistic_regression import OnlineLogisticRegression

# Train model
model = OnlineLogisticRegression(n_features=5)
for x, y in data_stream:
    model.update(x, y)

# Save checkpoint
checkpoint = {
    'model': model.get_state(),
    'hyperparams': {
        'n_features': model.n_features,
        'lr0': model.lr0,
        'lr_schedule': model.lr_schedule
    }
}
with open('checkpoint.json', 'w') as f:
    json.dump(checkpoint, f)

# Restore
with open('checkpoint.json') as f:
    checkpoint = json.load(f)

model = OnlineLogisticRegression(**checkpoint['hyperparams'])
model.set_state(checkpoint['model'])
```

---

##  Performance Characteristics

### Computational Complexity (per sample)

| Operation | Time | Space |
|-----------|------|-------|
| `predict()` | O(n_features) | O(n_features) |
| `update()` | O(n_features) | O(1) |
| `fit_transform()` | O(n_features) | O(n_features) |
| `evaluator.record()` | O(1) | O(1) |
| `get_metrics()` | O(W log W)* | O(W) |

*W = window size, log factor from AUC rank sorting

### Memory Footprint

- **OnlineLogisticRegression**: O(n_features) for weights + O(1) per step
- **OnlineScaler**: O(n_features) for mean and M2 arrays
- **SlidingWindowEvaluator**: O(window_size) for prediction buffer
- **StreamLoader**: O(chunk_size) for read buffer

**Total**: O(n_features + window_size + chunk_size) — independent of dataset size

### Numerical Stability

- Sigmoid overflow protection via logit clipping to [-500, 500]
- Binary cross-entropy clipping to [1e-15, 1-1e-15] prevents log(0)
- Welford's algorithm maintains precision for variance computation
- Epsilon guard in scaler prevents division by zero

---

##  When to Use Online Learning

###  Ideal Scenarios

- **Data arrives continuously** (e.g., sensor streams, financial ticks, user events)
- **Dataset exceeds available memory**
- **Low-latency predictions required** as data arrives
- **Distribution shifts over time** (concept drift)
- **Model must adapt to new patterns** without full retraining
- **Edge deployment** with computational constraints

###  Limitations

- May not reach batch model accuracy on stationary data
- Sensitive to learning rate and schedule selection
- Order-dependent (different orderings yield different results)
- No global optimization guarantees (local convergence only)

###  Practical Applications

- **Fraud Detection**: Real-time transaction scoring with adaptive thresholds
- **Network Intrusion Detection**: Evolving attack pattern recognition
- **Recommendation Systems**: Continuous adaptation to user preferences
- **IoT Sensor Monitoring**: Anomaly detection in streaming telemetry
- **Predictive Maintenance**: Equipment failure prediction from continuous metrics
- **Click-Through Rate Prediction**: Ad performance optimization with daily updates

---
### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatters
black core/ data/ evaluation/ drivers/ tests/
flake8 core/ data/ evaluation/ drivers/ tests/

# Type checking
mypy core/ --ignore-missing-imports
```

---

### Related Projects

- [River](https://riverml.xyz/) — Python library for online machine learning
- [Vowpal Wabbit](https://vowpalwabbit.org/) — High-performance online learning system
- [scikit-multiflow](https://scikit-multiflow.readthedocs.io/) — Data stream learning framework
- [MOA](https://moa.cms.waikato.ac.nz/) — Massive Online Analysis framework (Java)

---
