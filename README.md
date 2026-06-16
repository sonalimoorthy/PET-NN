# PET_NN_JAX

A JAX-based neural network surrogate for **Physiological Equivalent Temperature (PET)** prediction. This project replaces computationally expensive PET calculations with a lightweight neural network capable of producing highly accurate predictions in milliseconds, making it suitable for large-scale urban climate simulations and thermal comfort studies.

---

## Overview

PET is a widely used thermal comfort index that quantifies how humans perceive outdoor environmental conditions. Traditional PET computation requires iterative thermophysiological calculations, which can become computationally expensive when evaluating thousands of environmental scenarios.

This repository implements a neural network surrogate that learns the relationship between environmental variables and PET, enabling rapid inference while maintaining high accuracy.

---

## Features

* Fast PET prediction using JAX
* Fully-connected neural network architecture
* Supports batch inference
* Training and evaluation pipeline included
* Benchmarking against direct PET calculations
* Low prediction error across a wide environmental range
* Suitable for urban climate and digital twin applications

---

## Input Variables

The model predicts PET using the following meteorological parameters:

| Variable | Description              | Unit |
| -------- | ------------------------ | ---- |
| Tdb      | Dry-bulb air temperature | °C   |
| Tr       | Mean radiant temperature | °C   |
| RH       | Relative humidity        | %    |
| v        | Wind speed               | m/s  |

---

## Model Architecture

The surrogate model is implemented as a Multi-Layer Perceptron (MLP) using JAX.

### Architecture

```text
Input Layer (4 Features)
        ↓
Dense Layer
        ↓
ReLU
        ↓
Dense Layer
        ↓
ReLU
        ↓
...
        ↓
Output Layer (PET)
```

### Training

* Framework: JAX
* Optimizer: AdamW
* Loss Function: Mean Absolute Error (MAE)
* Batch Training
* Early Stopping
* Learning Rate Scheduling

---

## Performance

Typical benchmark results:

| Metric         | Value         |
| -------------- | ------------- |
| MAE            | ~0.06–0.08 °C |
| RMSE           | ~0.15–0.25 °C |
| P95 Error      | < 0.25 °C     |
| P99 Error      | < 1 °C        |
| Inference Time | Milliseconds  |

The model provides several orders of magnitude speedup compared to direct PET computation while maintaining excellent predictive accuracy.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PET_NN_JAX.git
cd PET_NN_JAX
```

Create a Python environment:

```bash
conda create -n pet_nn python=3.11
conda activate pet_nn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Launch the notebook:

```bash
jupyter notebook PET_NN_JAX.ipynb
```

The notebook includes:

* Dataset generation
* Feature preprocessing
* Model training
* Validation
* Benchmark evaluation
* Model export

---

## Inference Example

```python
import numpy as np

sample = np.array([
    [30.0, 40.0, 60.0, 1.5]
])

pet_prediction = model.predict(sample)

print(pet_prediction)
```

Input order:

```python
[Tdb, Tr, RH, v]
```

---

## Repository Structure

```text
PET_NN_JAX/
│
├── PET_NN_JAX.ipynb      # Training and evaluation notebook
├── models/               # Saved checkpoints
├── datasets/             # Training and testing datasets
├── benchmarks/           # Benchmark results
├── requirements.txt
└── README.md
```

---

## Applications

* Urban climate simulations
* Thermal comfort assessment
* Urban digital twins
* Building performance analysis
* Smart city applications
* Environmental modelling

---

## Limitations

* Performance may degrade for meteorological conditions far outside the training range.
* Extreme low-wind and high-radiation cases can produce larger prediction errors.
* Retraining may be required for substantially different climate regimes.

---

## Future Improvements

* Expanded training datasets
* Improved handling of extreme weather conditions
* Physics-informed neural networks
* Uncertainty quantification
* GPU-optimized deployment pipeline

---

## Citation

If you use this repository in your research, please cite the associated project or publication when available.

---
