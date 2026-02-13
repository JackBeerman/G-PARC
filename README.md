# G-PARC: Graph Physics-Aware Recurrent Convolutions

This repository contains the code for **G-PARC**, a physics-aware deep learning framework for forecasting complex spatiotemporal dynamics on unstructured meshes. G-PARC combines graph neural networks with numerical methods — including Moving Least Squares (MLS) differential operators and explicit time integrators — to learn physical dynamics directly from simulation data.

The framework is applied to multiple domains: elastoplastic impact dynamics, river flood forecasting, shock wave propagation, and 3D vortex shedding.

## Model Weights & Data

Trained model checkpoints, test datasets, and configuration files are hosted on Hugging Face:

**🤗 [huggingface.co/jacktbeerman/Gparc](https://huggingface.co/jacktbeerman/Gparc)**

The demo notebooks download these artifacts automatically.

## Repository Structure

```
G-PARC/
├── models/              # Model architectures (G-PARCv1, G-PARCv2, etc.)
├── differentiator/      # MLS-based differential operators & physics modules
├── integrator/          # Numerical time integration (Euler, Heun, RK4)
├── utilities/           # Feature extractors, SPADE fusion, training utils
├── data/                # Dataset classes & normalization
├── scripts/             # Training & evaluation scripts
├── demos/               # Demo notebooks for each dataset
├── MeshGraphNet/        # MeshGraphNet baseline implementation
├── FNOGNO/              # FNO-GNO baseline implementation
├── assets/              # GIFs and figures for README
└── requirements.txt
```

## Demonstrations

### White River Flood Forecasting 🌊

The model forecasts inundation area and water level changes for a flooding event on the White River.

![White River Flood Forecast](./assets/whiteriver_gparc.gif)

### Shock Tube Simulations 💥

Shock tube simulation results demonstrating the model's stability and accuracy under different conditions.

![Shock Tube](./assets/shocktube_total.gif)

### Elastoplastic Impact Dynamics 🔩

Side-by-side comparison of ground truth and model predictions on the PLAID elastoplastic benchmark, showing mesh deformation under high-velocity impact.

![Elastoplastic Deformation](./assets/elasto_deformed.gif)

## Getting Started

### Installation

```bash
git clone https://github.com/JackBeerman/G-PARC.git
cd G-PARC
pip install -r requirements.txt
```

**Note:** PyTorch and PyTorch Geometric should be installed separately based on your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) and [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Running the Demo Notebooks

The demo notebooks in `demos/` automatically download model weights and test data from Hugging Face:

```bash
cd demos/elastoplastic
jupyter notebook plaid_elastoplastic_demo.ipynb
```

No manual data setup required — all artifacts are fetched and cached on first run.

## Datasets

This work uses the following publicly available datasets:

**Elastoplastic Impact Dynamics** — 2D ElastoPlastoDynamics from the PLAID benchmark suite. High-velocity impact simulations on steel plates with nonlinear elastoplastic constitutive laws, solved with OpenRadioss on unstructured meshes.
> Casenave, F., Roynard, X., Staber, B., Piat, W., et al. "Physics-Learning AI Datamodel (PLAID) datasets: a collection of physics simulations for machine learning." *arXiv:2505.02974*, 2025. [[Paper]](https://arxiv.org/abs/2505.02974) [[Data (Zenodo)]](https://zenodo.org/records/15286369) [[HuggingFace]](https://huggingface.co/PLAID-datasets)

**River Flood Forecasting** — White River and Iowa River flood simulation data from HydroGraphNet. 2D shallow water equation solutions on unstructured meshes with varying hydrograph boundary conditions.
> Taghizadeh, M., Zandsalimi, Z., Nabian, M.A., Shafiee-Jood, M., & Alemazkoor, N. "Interpretable physics-informed graph neural networks for flood forecasting." *Computer-Aided Civil and Infrastructure Engineering*, 2025. [[Paper]](https://doi.org/10.1111/mice.13484)

## Architecture Overview

**G-PARCv2** (the primary architecture) consists of:

1. **Graph Convolution Layers** — extract spatial features from unstructured mesh data
2. **MLS Differential Operators** — compute physics-grounded gradients and Laplacians via Moving Least Squares
3. **SPADE Fusion** — combine learned features with differential quantities through spatially-adaptive normalization
4. **Numerical Integration** — advance the state forward in time using Euler, Heun, or RK4 schemes

## Citation

*Paper under review.*

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.