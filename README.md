# Parametric resonance in a spin-1/2 chain: dynamical effects of nontrivial topology

This repository contains the source code, data, and plotting scripts associated with the paper:
**"Parametric resonance in a spin-1/2 chain: dynamical effects of nontrivial topology"** by Mahmoud T. Elewa and M. I. Dykman, submitted to *Physical Review Letters*.

## Repository Structure

- **src/**: Contains the core Julia simulation codes.
  - `sudden_run.jl`: Simulates the sudden quench of the drive amplitude as described in the Supplemental Material of the paper
  - `adiabatic_run.jl`: Simulates the adiabatic evolution of the drive amplitude as described in the Supplemental Material of the paper
  - `adiabatic_run_mu.jl`: Simulates the adiabatic evolution on the hysteresis path as described in the Supplemental Material of the paper
- **notebooks/**: Jupyter notebooks used to generate the figures in the paper.
  - Python notebooks: Used for correlation plots and hysteresis.
  - Julia notebooks: Used for sudden scaling analysis with different N.
- **data/**: Raw simulation data in .txt format. Each subfolder contains a `1Description.txt` file explaining the specific dataset.
- **figures/**: The output figures presented in the manuscript and the Supplemental Material.

## How to Run

### Prerequisites

To use these codes, you will need the following software and libraries installed.

#### 1. Julia
**Version:** 1.11 (or higher)

The simulation codes rely on the following packages:
* `ITensors.jl` (and `ITensorMPS`)
* `JLD2.jl`
* `Plots.jl`
* `LaTeXStrings.jl`
* `QuadGK.jl`
* Standard Libraries: `LinearAlgebra`, `Statistics`, `DelimitedFiles`, `Base.Threads`

**Installation:**
Run the following command in the Julia REPL:
```julia
import Pkg
Pkg.add(["ITensors", "JLD2", "Plots", "LaTeXStrings", "QuadGK"])

#### 2. Python
**Version:** 3.13 (or higher)

The plotting notebooks require the following libraries:
* `numpy`
* `matplotlib`
* `scipy`
* `jupyter`

**Installation:**
Run the following command in your terminal:
```bash
pip install numpy matplotlib scipy jupyter

### Instructions

1. **Clone this repository** to your local machine.

2. **Run Simulations (Julia):**
   Navigate to the `src/` folder. To reproduce the simulation data, run the files using Julia.
   *Tip: To enable multi-threading for faster execution, start Julia with the `-t` flag.*
   ```bash
   julia -t 4 src/sudden_run.jl
   
3. **Reproduce Plots:**
   Navigate to the `notebooks/` folder and launch Jupyter.
   ```bash
   jupyter notebook
      Open the desired notebook (e.g., for correlation plots) and run all cells.

## Citation

If you use this code or data, please cite our paper:
https://arxiv.org/abs/2511.10891