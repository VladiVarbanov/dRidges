# dRidges

Modular framework for STEM microscopy defect detection and analysis using computer vision and neural networks.

> Experimental research framework under active development.

---

## Overview

dRidges is a modular research-oriented framework focused on defect detection and analysis in STEM microscopy data.

The project combines:
- classical computer vision techniques,
- neural network pipelines,
- configurable preprocessing stages,
- dataset conversion utilities,
- and experimental hybrid CV/NN workflows.

The current development focuses on:
- TorchVision-based object detection,
- modular preprocessing pipelines,
- annotation conversion and validation,
- dataset caching,
- and reproducible experimentation workflows.

---

## Current Features

- Modular multi-channel preprocessing pipeline
- TorchVision dataset integration
- Bounding box conversion utilities
- Detection target generation
- Dataset split handling
- Intermediate caching system
- Experimental detection workflow infrastructure

---

## Planned Features

- Hybrid CV + NN defect analysis workflows
- Orientation normalization / canonicalization
- PCA-assisted ROI alignment
- Experimental descriptor pipelines
- Advanced visualization and debugging tools
- Local AI-assisted workflow orchestration
- Distributed preprocessing and inference support

---

## Project Structure

```text
src/                    # Core source code
notebooks/              # Research and exploratory notebooks
docs/                   # Documentation and diagrams
configs/                # Configuration files
tests/                  # Smoke tests and validation scripts
outputs/                # Generated outputs and predictions
