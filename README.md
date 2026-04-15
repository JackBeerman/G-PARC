# G-PARC: Graph-Physics Aware Recurrent Convolutional Neural Networks

G-PARC is a physics-aware deep learning framework for forecasting complex spatiotemporal dynamics on **unstructured meshes**. It combines graph neural networks with numerical methods — Moving Least Squares (MLS) differential operators and explicit time integrators — to learn physical dynamics directly from simulation data.

The framework is validated across three challenging domains: **elastoplastic dynamics**, **planar shock wave flows**, and **river flood forecasting**.

*Paper under review.*

---

## Highlights

- **MLS differential operators** (gradient, Laplacian, strain) computed directly on unstructured meshes
- **Temporal conditioning** for variable-timestep generalization — a single trained model handles arbitrary Δt at inference
- **Numerical time integration** (Euler, Heun, RK4) replacing learned integrators for physical consistency
- **604–2,382× throughput advantage** over neural operator baselines (GINO/GNO)
---

## Model Weights & Data

Trained model checkpoints, test datasets, and configuration files are hosted on Hugging Face:

**🤗 [huggingface.co/jacktbeerman/Gparc](https://huggingface.co/jacktbeerman/Gparc)**

The demo notebooks download these artifacts automatically — no manual data setup required.

---

## Repository Structure

```
G-PARC/
├── models/              # Model architectures (G-PARC, G-PARC (w/o MLS), baselines)
├── differentiator/      # MLS differential operators & physics modules
├── integrator/          # Numerical time integration (Euler, Heun, RK4)
├── utilities/           # Feature extractors, SPADE fusion, training utilities
├── data/                # Dataset classes & normalization
├── scripts/             # Training & evaluation scripts per domain
├── demos/               # Demo notebooks with Hugging Face auto-download
├── tests/               # Unit tests for operators and per-domain models
├── visualizations/      # Visualization, metrics, and comparison utilities
├── assets/              # GIFs for README
└── requirements.txt
```

---

## Demonstrations

### Elastoplastic Impact Dynamics

Side-by-side comparison of ground truth and G-PARC predictions on the PLAID elastoplastic benchmark, showing mesh deformation.

![Elastoplastic Deformation](./assets/elasto_deformed.gif)

### Shock Tube Simulations

Compressible Euler equation solutions demonstrating stability and accuracy across varying initial pressure ratios and timestep sizes.

![Shock Tube](./assets/shocktube_total.gif)

### White River Flood Forecasting

Flood inundation forecasting on unstructured HEC-RAS meshes, predicting water surface elevation and depth evolution.

![White River Flood Forecast](./assets/whiteriver_gparc.gif)

---

## Architecture Overview

**G-PARC** (the primary architecture) follows a modular Differentiate → Integrate design:

1. **Graph Convolution Layers** — extract spatial features from unstructured mesh node/edge data using GATConv
2. **MLS Differential Operators** — compute physics-grounded spatial derivatives (gradients, Laplacians, strain rates) via Moving Least Squares on the mesh stencil
3. **SPADE Fusion** — combine learned GNN features with MLS differential quantities through spatially-adaptive normalization
4. **FiLM Conditioning** — modulate learned representations with simulation parameters (pressure ratio, Δt) for variable-condition generalization
5. **Numerical Integration** — advance the state forward in time using Euler, Heun, or RK4 schemes

**G-PARC (w/o MLS)** uses a learned GNN integrator (IntegralGNN) instead of numerical schemes, serving as an ablation baseline.

---

## Getting Started

### Installation

```bash
git clone https://github.com/JackBeerman/G-PARC.git
cd G-PARC
pip install -r requirements.txt
```

> **Note:** PyTorch and PyTorch Geometric should be installed separately based on your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Running Demos

The demo notebooks in `demos/` automatically download model weights and test data from Hugging Face:

```bash
cd demos/elastoplasto
jupyter notebook plaid_elastoplastic_demo.ipynb
```

### Running Tests

```bash
python tests/run_all.py
```

This runs operator-level and per-domain integration tests to verify model correctness.

---

## Datasets

### Elastoplastic Impact Dynamics (PLAID)

2D elastoplastodynamics from the PLAID benchmark suite — high-velocity impact simulations on steel plates with nonlinear elastoplastic constitutive laws, solved with OpenRadioss on unstructured meshes.

> Casenave, F., Roynard, X., Staber, B., Piat, W., et al. "Physics-Learning AI Datamodel (PLAID) datasets: a collection of physics simulations for machine learning." *arXiv:2505.02974*, 2025. [[Paper]](https://arxiv.org/abs/2505.02974) [[Data (Zenodo)]](https://zenodo.org/records/15286369) [[HuggingFace]](https://huggingface.co/PLAID-datasets)

### Compressible Shock Tube

1D compressible shock tube simulations on 2D domain solving the Euler equations with varying initial pressure and density ratios. Simulation data generated using the high-order finite-volume combustion solver of Gao et al.

> Gao, X., Owen, L. D., & Guzik, S. M. "A high-order finite-volume method for combustion." In *54th AIAA Aerospace Sciences Meeting*, p. 1808, 2016. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0898122116304588)

### River Flood Forecasting (HydroGraphNet)

White River flood simulation data from HydroGraphNet — 2D shallow water equation solutions on unstructured meshes with varying hydrograph boundary conditions.

> Taghizadeh, M., Zandsalimi, Z., Nabian, M.A., Shafiee-Jood, M., & Alemazkoor, N. "Interpretable physics-informed graph neural networks for flood forecasting." *Computer-Aided Civil and Infrastructure Engineering*, 2025. [[Paper]](https://doi.org/10.1111/mice.13484)

---

## Baselines

The repository includes implementations of the following baseline models used for comparison:

- **MeshGraphNet** — encoder-processor-decoder GNN from NVIDIA PhysicsNeMo ([Pfaff et al., 2021](https://arxiv.org/abs/2010.03409); [PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html))
- **MeshGraphKAN** — MeshGraphNet variant with Fourier KAN layers replacing MLPs, reimplemented from NVIDIA PhysicsNeMo
- **GraphSAGE** — sampling-based GNN baseline ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216))
- **G-PARC (w/o MLS)** — ablation using learned integration instead of numerical schemes

---

## Citation

```bibtex
@article{beerman2026gparc,
  title={G-PARC: Graph-Physics Aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics on Unstructured Meshes},
  author={Beerman, Jack T. and Abele, Tyler J. and Taghizadeh, Mehdi and Davis, Andrew and Gray, Zo{\"e} J. and Alemazkoor, Negin and Gao, Xifeng and Udaykumar, H. S. and Baek, Stephen S.},
  journal={Under review},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
