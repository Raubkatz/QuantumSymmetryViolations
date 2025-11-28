# QuantumSymmetryViolations




# Reconstructing Quantum Noise from Symmetry-Violating Generators  
## A Complete Framework for Single-Qubit Process Tomography, Noise Modeling, and Evaluation

## Overview
This repository provides an end-to-end framework for reconstructing and analyzing single-qubit noise using symmetry-violating generator perturbations.  
It combines:

- **Analytic and shot-based QPT-12 tomography** (custom implementation).
- **Direct Qiskit Experiments process tomography** (hardware and simulator).
- **Generator-level noise modeling** using additive Δ perturbations inside the exponent.
- **Genetic-algorithm-based parameter fitting** of Δ maps.
- **Hermitian and non-Hermitian random noise sweeps** for statistical benchmarking.
- **Evaluation pipelines** for ideal, noisy, and hardware-targeted PTMs.
- **Aggregation tools** for curated hardware tomography datasets.

The workflow is aligned with the conventions used in Qiskit Experiments while remaining fully explicit through NumPy/SciPy.  
All PTMs follow the Pauli basis ordering **{I, X, Y, Z}**.

## Features
- **Analytic Tomography Core**  
  A complete from-scratch implementation of state and process tomography, including:
  - Gate construction via matrix exponentials with additive generator noise (Δ).  
    :contentReference[oaicite:0]{index=0}
  - Exact PTM mapping in the Heisenberg picture.
  - Affine minimal-schedule tomography with finite-shot sampling.
  - Optional amplitude- and phase-damping CPTP channels.
  - Canonical preparation and Pauli measurement model.

- **Process Tomography on Hardware**  
  A Qiskit-based script for running a full single-qubit X-gate process tomography on IBM Quantum hardware or simulator, saving:  
  - Raw Qiskit Experiments results  
  - PTMs  
  - Logs and artefacts per run  
    :contentReference[oaicite:1]{index=1}

- **Genetic Algorithm Fitting of Symmetry-Violating Δ Maps**  
  A GA minimizes the difference between QPT-12 PTMs and a hardware-target PTM.  
  It supports Hermitian and general complex Δ maps and handles trace-preservation constraints.  
    :contentReference[oaicite:2]{index=2}

- **Evaluation of Noisy Variants Using QPT-12**  
  A dedicated process-only evaluator runs analytic and shot-based QPT-12 over ideal operations and noisy Δ-variants.  
  It generates PTM means, standard deviations, heatmaps, and report files.  
    :contentReference[oaicite:3]{index=3}

- **Hermitian / Non-Hermitian Noise Sweeps**  
  A full statistical sweep generator constructs random Δ families and evaluates mean/std PTMs and z-scores relative to hardware targets.  
    :contentReference[oaicite:4]{index=4}

- **Hardware Aggregation Pipeline**  
  A tool for aggregating curated process tomography results from `unique_process_tomos/` into:  
  - Mean PTM  
  - Standard deviation  
  - Deviation from the ideal X PTM  
    :contentReference[oaicite:5]{index=5}

## Repository Structure

```
📂 quantum_noise_reconstruction/

├── class_tomography_final.py
│   ├── Analytic and shot-based tomography (QPT-12 strict)
│   ├── Gate generators and matrix-exponential construction
│   ├── Pauli basis, state prep, measurement model
│   ├── Kraus noise channels (optional)
│   └── PTM reconstruction utilities

├── 00_final_hardware_experiment_xgate_process_tomography.py
│   ├── Runs Qiskit Experiments process tomography on hardware/simulator
│   ├── Saves PTMs, logs, run metadata
│   └── Provides target PTM datasets for downstream analysis

├── 01_final_analyze_tomography_results.py
│   └── High-level analyzer for tomography outputs (PTM inspection, summaries)

├── 02_final_genetic_algorithm.py
│   ├── GA fitting of Δ-maps for gates {Rz, SX, X}
│   ├── Cost is QPT-12 PTM deviation from hardware mean
│   ├── Supports TP constraints and error amplification
│   └── Produces optimized Δ families and fitted PTMs

├── 03_final_eval_qiskit_noisy_matrices.py
│   ├── QPT-12 evaluation of ideal and noisy Δ-based models
│   ├── Multi-shot evaluation (analytic + sampled)
│   ├── Heatmap generation and mean/std PTM summaries
│   └── Report generation

├── 04_final_hermitian_noise_analysis.py
│   ├── Hermitian and non-Hermitian random Δ sweeps
│   ├── Mean/std PTM for each noise family
│   ├── Hardware target comparison
│   ├── Z-score matrices
│   └── Summary text and matrix outputs

├── aggregate_unique_tomos.py
│   ├── Aggregates curated hardware tomography results
│   ├── Produces R_avg, R_std, R_err
│   ├── Manifest JSON/CSV
│   └── Mean±std tables and summaries

└── README.md
```

## Running the Code

### 1. Generate or Collect Hardware PTMs
Run process tomography on hardware or a simulator:

```
python 00_final_hardware_experiment_xgate_process_tomography.py
```

This creates one directory per run, containing PTMs and metadata.

### 2. Aggregate Hardware Tomography
Create mean and standard-deviation PTMs across curated runs:

```
python aggregate_unique_tomos.py
```

Outputs include:
- `R_avg.npy`, `R_std.npy`, `R_err.npy`
- `R_mean_std.npy`
- CSV versions
- Manifest and summaries

### 3. GA-Based Fitting of Δ Maps
Fit generator-level perturbations to match hardware PTM:

```
python 02_final_genetic_algorithm.py
```

This:
- Initializes the GA population  
- Evolves Δ parameters for gates (Rz, SX, X)  
- Evaluates cost through QPT-12 reconstruction  
- Saves the best Δ map and fitted PTMs  

### 4. Evaluate Ideal and Noisy Models via QPT-12
Run analytic and finite-shots QPT-12 evaluation:

```
python 03_final_eval_qiskit_noisy_matrices.py
```

Outputs include:
- Ideal analytic PTM
- Ideal sampled PTM mean/std
- Noisy variant PTMs
- Heatmaps and text reports

### 5. Hermitian / Non-Hermitian Noise Sweeps
Evaluate random Δ-generated noise families:

```
python 04_final_hermitian_noise_analysis.py
```

Outputs include:
- Mean and std PTMs for each family
- Hardware vs. synthetic z-score matrices
- Plots and reports

## Requirements

```
python==3.11.13
numpy==2.3.0
scipy==1.15.3
matplotlib==3.10.3
qiskit==2.0.3
qiskit-aer==0.17.1
qiskit-experiments==0.10.0
qiskit-ibm-runtime==0.40.1

```

## Citation and Context
This repository implements the experimental and analytic pipeline used in the study:

**“Reconstructing Quantum Noise from Symmetry-Violating Generators”**  
Raubitzek et al., 2025  

It reproduces the full tomography workflow used in the manuscript and provides all tools required to evaluate generator-level noise models against real hardware process tomographies.


