# eCoMFA

## Overview
The **eCoMFA** (electronic Comparative Molecular Field Analysis) is a computational method used for quantitative structure-selectivity relationship (QSSR) analysis. This repository contains an implementation of the CoMFA model, designed for analyzing molecular data and predicting reaction selectivity based on 3D electronic molecular field properties.

## Features
- Implementation of the CoMFA model for QSSR analysis
- Machine learning algorithms for predictive modeling

## Installation
To install the necessary dependencies, run:

```bash
conda env export -n my_env > environment.yml
```

## Usage
### 1. Prepare Molecular Data
Prepare your dataset containing molecular structures and selectivity. XLSX files of the dataset can also be used.

### 2. Run DFT Calculation
Use the following command to execute DFT calculation:

```bash
python lib/calc_mol.py
```

### 3. Run the Feature Calculation
Use the following command to execute the feature calculation:

```bash
python lib/calc_grid_parallel.py
```

### 4. Run the Regression
Use the following command to execute the regression:

```bash
python lib/regression_parallel.py
```

### 5. Run the make graph and Evaluation
Use the following command to make graph and evaluation:

```bash
python lib/graph.py
```

### 3. Analyze Results
The output will contain predicted selectivity values and model performance metrics.

## Dataset Requirements
- Molecular structures in SMILES strings and reaction temperature
- Corresponding selectivity values for supervised learning

## Dependencies
- Python (4.0 is recommended)
- Gaussian
- Psi4
- RDKit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## License


## Authors
Developed by the **POC Lab** team.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a Pull Request

## Contact
For inquiries or support, please contact the POC Lab team.

