# ocpy — O‑C (Observed – Calculated) Analysis for Astronomers

![OC_PY](https://github.com/mshemuni/oc_py/actions/workflows/OC_PY.yml/badge.svg)
![OC_PY](https://img.shields.io/badge/coverage-60%25-31c553)
[![Documentation Status](https://readthedocs.org/projects/oc-py/badge/?version=latest)](https://oc-py.readthedocs.io/en/latest/?badge=latest)
![OC_PY](https://img.shields.io/badge/Win-%E2%9C%93-f5f5f5?logo=windows11)
![OC_PY](https://img.shields.io/badge/Ubuntu-%E2%9C%93-e95420?logo=Ubuntu)
![OC_PY](https://img.shields.io/badge/MacOS-%E2%9C%93-dadada?logo=macos)
![OC_PY](https://img.shields.io/badge/Python-%203.11,%203.12,%203.13,%203.14-3776ab?logo=python)
![OC_PY](https://img.shields.io/badge/LIC-GNU/GPL%20V3-a32d2a?logo=GNU)

**Documentation**: [oc-py.readthedocs.io](https://oc-py.readthedocs.io/en/latest/)

`ocpy` is a robust Python library designed for **O‑C (Observed – Calculated) analysis**, a core technique in astronomy for studying period variations in binary systems, transiting exoplanets, and pulsating stars.

---

## Key Features

*   **Data Handling**: Seamlessly load timing data from Excel or CSV files.
*   **Weighted Analysis**: Automatic weight calculation based on observational uncertainties.
*   **Model Components**: Flexible building blocks including `Linear`, `Quadratic`, `Sinusoidal`, and `Keplerian` (Light-Time Effect) models.
*   **Dual Fitting Engines**:
    *   **Frequentist (LMFit)**: Fast non-linear least-squares optimization.
    *   **Bayesian (PyMC)**: Full posterior inference with MCMC sampling for robust uncertainty quantification.
*   **Visualization**: Specialized O‑C plotting with model overlays and residual analysis.

---

## Quick Start

### 1. Installation

#### Via pip (Recommended)
```bash
pip install oc_py
```

#### From Source
```bash
# Clone the repository
git clone https://github.com/mshemuni/oc_py.git
cd oc_py

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Workflow

#### Step 1: Load and Preprocess Data
```python
from ocpy.data import Data

# Load data from file
data = Data.from_file("my_data.xlsx")

# Calculate statistical weights (w = 1/sigma^2)
data = data.calculate_weights()
```

#### Step 2: Calculate O-C Residuals
```python
# Compute O-C relative to a reference ephemeris
oc = data.calculate_oc(reference_minimum=2450000.5, reference_period=0.5, model_type="pymc")
```

#### Step 3: Define Model and Fit (Bayesian Example)
```python
from ocpy.oc import Linear, Quadratic, Parameter

# Define priors for model components
lin = Linear(
    a=Parameter(value=0.0, std=1e-5, fixed=False),
    b=Parameter(value=0.0, std=1e-3, fixed=False)
)
quad = Quadratic(
    q=Parameter(value=0.0, std=1e-9, fixed=False)
)

# Fit model using PyMC (MCMC sampling)
res = oc.fit([lin, quad], draws=2000, tune=1000, chains=4)

# Visualize the result
oc.plot(model=[lin, quad])
```

---

## Bayesian Workflow Details

`ocpy` leverages **PyMC** for sophisticated Bayesian inference. The standard workflow follows:

1.  **Prior Specification**: Uses the `Parameter` class to define initial values, standard deviations, and optional bounds (Truncated Normal).
2.  **Likelihood Selection**: Assumes a **Gaussian Likelihood** where residuals are normally distributed around the composite model.
3.  **Composite Modeling**: Individual components are automatically summed to create the total predicted O‑C signal.
4.  **Inference**: Utilizes the **NUTS (No-U-Turn Sampler)** to efficiently sample the posterior distribution.

---

## Project Structure

*   `src/ocpy/data.py`: Data container and O‑C arithmetic.
*   `src/ocpy/oc.py`: Core model definitions and base classes.
*   `src/ocpy/oc_pymc.py`: Bayesian inference implementation.
*   `src/ocpy/oc_lmfit.py`: Frequentist optimization implementation.
*   `docs/exe/`: Comprehensive examples (e.g., NY Vir, DD CrB datasets).

---

## Contributing

Contributions are welcome! Please ensure all code passes the `mypy` and `flake8` checks, and add unit tests for new model components in the `tests/` directory.

---

## Feedback & Support

Found a bug or need help? [Open an issue](https://github.com/mshemuni/oc_py/issues) here on GitHub and include:
*   A clear description of the problem or feature request.
*   A minimal reproducible example (code snippet and sample data if possible).
*   Your environment details (Python version, OS).

---

## License

Distributed under the **GPL-3.0 License**. See `LICENSE` for more information.
