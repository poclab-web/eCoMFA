# eCoMFA

![](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![](https://img.shields.io/badge/License-MIT-orange)

## Overview
This repository implements an **electronic CoMFA (eCoMFA)** workflow for quantitative structure-selectivity relationship (QSSR) analysis.

The pipeline:
1. prepares molecular datasets from Excel files,
2. runs conformer and quantum-chemical calculations,
3. converts volumetric outputs into CoMFA grid features,
4. trains regression models (Lasso/Ridge/ElasticNet/PLS), and
5. generates evaluation tables and publication-ready figures.

## Repository Structure
- `lib/dataset.py`: build curated training/test datasets from raw Excel files.
- `lib/calc_mol.py`: generate conformers, run Gaussian/Psi4, and save `geom/Dt/ESP` outputs per molecule.
- `lib/calc_grid.py`: aggregate cube/log outputs into folded/unfolded grid features and write `.pkl` datasets.
- `lib/regression.py`: run model sweeps in parallel and save predictions/coefficients.
- `lib/graph_main.py`: evaluate model performance and generate core regression plots/tables.
- `lib/graph_cont.py`: generate contribution-space plots (`electronic_cont` vs `electrostatic_cont`).
- `lib/render_moleculer.py`: helper functions for 3D molecular/grid rendering.
- `lib/definition_grid.ipynb`, `lib/ts_conformer_calc.ipynb`: exploratory notebooks.
- `eCoMFA_all_calculation.ipynb`: end-to-end notebook that runs the full workflow step by step.
- `dataset/`: raw source datasets.
- `results/`: processed datasets and model outputs.

## Requirements
### Software
- Python (Conda recommended)
- Gaussian (`g16` available in shell)
- Psi4

### Python packages
See `environment.yml`.

## Environment Setup
Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

If needed, update `<env-name>` to the `name:` defined in your `environment.yml`.

## Workflow
Run from repository root.

### Notebook (all steps in one place)
Open and run:
```bash
jupyter notebook eCoMFA_all_calculation.ipynb
```
The notebook executes the full pipeline in order:
`lib/dataset.py` -> `lib/calc_mol.py` -> `lib/calc_grid.py` -> `lib/regression.py` -> `lib/graph_main.py`.
For contribution-space plots, run `python lib/graph_cont.py` separately after the notebook.

### 1. Build dataset files
```bash
python lib/dataset.py
```
Inputs: `dataset/*.xlsx`  
Outputs: `results/CBS.xlsx`, `results/DIP.xlsx`, `results/alpine_borane.xlsx`, `results/mol_list.xlsx`

### 2. Run molecular quantum calculations
```bash
python lib/calc_mol.py
```
Default output root is `~/CoMFA_calc` (configurable in `lib/calc_mol.py`).  
Each molecule directory gets a `done` file when calculations complete successfully.

### 3. Build CoMFA grid features
```bash
python lib/calc_grid.py
```
Reads `results/*.xlsx` and `~/CoMFA_calc/<InChIKey>/...`, then writes `results/*.pkl`.

### 4. Train regressions
```bash
python lib/regression.py
```
Writes regression outputs such as:
- `results/*_regression.pkl`
- `results/*_regression.csv`

### 5. Evaluate and visualize
```bash
python lib/graph_main.py
python lib/graph_cont.py
```
Produces summary CSV/XLSX and figures (e.g., `results/regression*.png`, `results/cont_*.png`, `results/results_with_rmse.png`).

## Configuration Notes
Key runtime settings are centralized near the top of scripts:
- `lib/calc_mol.py`: `NUM_THREADS`, `MEMORY_GB`, `OUTPUT_ROOT`
- `lib/calc_grid.py`: `NUM_WORKERS`, `CALC_ROOT`
- `lib/regression.py`: `NUM_WORKERS`
- `lib/graph_main.py` / `lib/graph_cont.py`: dataset and output root constants

## Citation
Sakaguchi, Daimon, Masaki Shimono, and Hiroaki Gotoh.  
"Analysis of Asymmetric Reduction of Ketones Using Three-Dimensional Electronic States."  
*The Journal of Physical Chemistry A* 129.39 (2025): 8945-8958.  
https://doi.org/10.1021/acs.jpca.5c03510

## License
This project is distributed under the MIT License. See `LICENSE.txt`.

## Contact
POC Lab (Hiroaki Gotoh): gotoh-hiroaki-yw[at]ynu.ac.jp
