# Protocol for Predicting Gibbs Energies and Inferring Thermodynamic Directions in Genome-Scale Metabolic Models Using dGbyG and ThermoInfer
---

## Summary

Integrating reaction thermodynamics is essential for refining constraint-based metabolic models. This repository provides the computational protocol coupling the **dGbyG** package with **ThermoInfer** for thermodynamic feasibility inference. The protocol describes the initial steps for setting up the computational environment and includes a worked example using the Yeast genome-scale metabolic model (GEM) to demonstrate how thermodynamic estimates can be used to evaluate reaction directionality.

![Graphical Abstract](GA.tif)

---

## Repository Contents

| File | Description |
|------|-------------|
| `workflow_yeast.ipynb` | Main Jupyter notebook demonstrating the full workflow |
| `run_tfba_yeast.py` | Python script for running thermodynamic flux balance analysis (TFBA) |
| `yeast-GEM.xml` | Yeast genome-scale metabolic model (GEM) in SBML format |
| `Yeast9_standard_dGr_dGbyG.csv` | Standard Gibbs free energy predictions from dGbyG for yeast reactions |
| `Yeast9_Directionality_TFBA.csv` | Inferred thermodynamic directionality results from ThermoInfer |
| `Yeast9_candidate_inconsistent_reactions.csv` | Candidate reactions with thermodynamically inconsistent directionality |

---

## Before You Begin

Reaction thermodynamics provides a physical basis for determining biochemical directionality. For a given metabolic reaction, its feasible direction is governed by the actual Gibbs free energy change (ΔrG), which depends on both standard thermodynamic properties (ΔrG°) and metabolite concentrations.

**dGbyG** uses graph neural networks (GNNs) to predict ΔrG°. 
**ThermoInfer**  uses ΔrG° estimates within a GEM to evaluate reaction directionality via thermodynamic flux balance analysis (TFBA).

### System Requirements

Users should run this protocol on a **Linux system** with **Conda** installed, ensuring sufficient CPU cores, memory, and a valid **Gurobi license** are available.

---

## Environment Setup

### 1. Confirm Git and Git LFS Availability

```bash
git --version
git lfs version
```

If Git LFS is not available:
```bash
conda install -c conda-forge git-lfs
git lfs install
```

> **CRITICAL:** Git LFS must be initialized before cloning repositories that contain Git LFS-managed files.

### 2. Download Repositories

```bash
mkdir -p ~/dgbyg_thermoinfer_protocol
cd ~/dgbyg_thermoinfer_protocol

# Clone dGbyG
git clone https://github.com/f-wc/dGbyG.git
cd ~/dgbyg_thermoinfer_protocol/dGbyG
git checkout feature/initial
git lfs pull

# Clone ThermoInfer
cd ~/dgbyg_thermoinfer_protocol
git clone https://gitee.com/f-wc/ThermoInfer.git
```

### 3. Create and Activate Conda Environment

```bash
cd ~/dgbyg_thermoinfer_protocol/dGbyG
conda env create -f environment.yml -n dgbyg-thermoinfer
conda activate dgbyg-thermoinfer

# Install additional packages
pip install libChEBIpy numpyarray-to-latex
```

### 4. Install Gurobi

```bash
conda install -c gurobi gurobi -y
```

> **CRITICAL:** Prepare a valid Gurobi license before running ThermoInfer. Academic users can obtain a free license from the [Gurobi User Portal](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Usage

Follow the step-by-step workflow in `workflow_yeast.ipynb`, which demonstrates:
1. Predicting standard Gibbs free energies using dGbyG
2. Running thermodynamic flux balance analysis (TFBA) with ThermoInfer
3. Identifying reactions with inconsistent thermodynamic directionality in the Yeast GEM

---

## Citation

If you use this protocol, please cite the associated study.
Wenchao Fan, Yonghong Hao, Xiangyu Hou, Chuyun Ding, Dan Huang, Weiyan Zheng, Ziwei Dai. Unraveling principles of thermodynamics for genome-scale metabolic networks using graph neural networks. Cell systems, 16(10), 101393.
---

