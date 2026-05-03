# Auxiliary Objectives for Video Joint-Embedding Predictive Architectures
Report : https://tinidrop.com/s/wdiw70lg

A systematic empirical study of auxiliary objectives for V-JEPA, evaluating twelve variants across three benchmarks.

## Key Finding

**Motion-Guided Masking** is the only auxiliary objective that achieves universal improvement across all benchmarks, without sacrificing motion for appearance or vice versa.

| Method | Diving-48 | ImageNet-100 | SSv2 |
|--------|-----------|--------------|------|
| Baseline V-JEPA | 8.38 | 12.02 | 2.07 |
| **Motion-Guided** | **8.68** (+0.30) | **12.16** (+0.14) | **3.45** (+1.38) |
| Kinematic-Accel | 5.79 (-2.59) | **13.74** (+1.72) | 3.18 (+1.11) |

All kinematic variants improve appearance (ImageNet-100) but degrade fine-grained action recognition (Diving-48), revealing **capacity interference** in single-latent JEPA architectures.

## Repository Structure

```
├── src/
│   ├── training/
│   │   ├── train.py           # Core training loop with all 12 auxiliary objectives
│   │   ├── utils.py           # Optimizer and model initialization
│   │   ├── transforms.py      # Data augmentation pipeline
│   │   └── wrappers.py        # Model wrappers
│   └── evaluation/
│       ├── eval_diving48.py    # Diving-48 frozen-encoder evaluation
│       ├── eval_imagenet100.py # ImageNet-100 linear probe evaluation
│       ├── eval_ssv2.py        # Something-Something V2 evaluation
│       ├── eval_diving48_tap.py
│       └── eval_diving48_full_tokens.py
├── configs/                    # Training configurations for all 14 methods
└── results/                    # Evaluation results (JSON) for all experiments
```

## Prerequisites

This code builds on top of the [V-JEPA](https://github.com/facebookresearch/jepa) codebase. To reproduce:

1. Clone the V-JEPA repository and install its dependencies
2. Copy `src/training/` files into `app/vjepa_2_1/` in the V-JEPA repo
3. Place configs in the V-JEPA configs directory
4. Prepare UCF-101 dataset

## Methods Studied

| Category | Methods |
|----------|---------|
| **Kinematic Regularization** | L1 (λ=1.0), L1 (λ=0.1), Accel, Huber, Split, Anneal |
| **Motion-Guided Masking** | Motion-biased mask sampling |
| **Anti-Collapse (SIGReg)** | SIGReg + EMA, SIGReg without EMA |
| **Physics-Inspired** | Hamiltonian-JEPA, Velocity-Gated JEPA |
| **Future Prediction** | Future-Predictive, Motion-Future |

## Experimental Setup

- **Encoder**: ViT-Base (86M parameters)
- **Pretraining**: UCF-101 (13,320 videos), 100 epochs
- **Hardware**: 4× NVIDIA A100, ~8 hours per run
- **Evaluation**: Frozen encoder with linear/attentive probes on Diving-48, ImageNet-100, SSv2

## References

- V-JEPA: Bardes et al., arXiv:2404.08471, 2024
- V-JEPA 2.1: Assran et al., arXiv:2603.14482, 2025
- LeJEPA / SIGReg: Maes et al., arXiv:2511.08544, 2025

## Author

Santosh Premi Adhikari — Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany
