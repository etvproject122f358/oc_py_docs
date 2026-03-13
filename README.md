# ocpy — O‑C (Observed – Calculated) Analysis for Astronomers

![OC\_PY](https://github.com/mshemuni/oc_py/actions/workflows/OC_PY.yml/badge.svg)
![OC\_PY](https://img.shields.io/badge/coverage-60%25-31c553)
[![Documentation Status](https://readthedocs.org/projects/oc-py/badge/?version=latest)](https://oc-py.readthedocs.io/en/latest/?badge=latest)
![OC\_PY](https://img.shields.io/badge/Win-%E2%9C%93-f5f5f5?logo=windows11)
![OC\_PY](https://img.shields.io/badge/Ubuntu-%E2%9C%93-e95420?logo=Ubuntu)
![OC\_PY](https://img.shields.io/badge/MacOS-%E2%9C%93-dadada?logo=macos)
![OC\_PY](https://img.shields.io/badge/Python-%203.11,%203.12,%203.13,%203.14-3776ab?logo=python)
![OC\_PY](https://img.shields.io/badge/LIC-GNU/GPL%20V3-a32d2a?logo=GNU)

**Documentation**: [oc-py.readthedocs.io](https://oc-py.readthedocs.io/en/latest/)

`ocpy` is a Python library for **O‑C (Observed – Calculated) analysis**, a core technique in astronomy used to study period variations in binary systems, pulsating stars, and transiting exoplanets.

---

## Key Features

* **Data Handling**: Load timing data from CSV or Excel effortlessly.
* **Weighted Analysis**: Automatic weight calculation based on observational uncertainties.
* **Flexible Model Components**:

  * `Linear`, `Quadratic`
  * `Sinusoidal`
  * `Keplerian` (Light-Time Effect, LiTE)
* **Dual Fitting Engines**:

  * **Frequentist (LMFit)** – Fast non-linear least-squares fitting.
  * **Bayesian (PyMC)** – Full posterior inference with MCMC for robust uncertainty estimates.
* **Advanced Visualization**:

  * O‑C data with error bars and labels.
  * Model overlays with residual plots.
  * Component-wise visualization.
  * Posterior median and uncertainty bands (1σ).
  * Trace and corner plots (`arviz` or `corner` style).
* **Compatibility**: Works with pandas DataFrames for input data.

---

## Installation

```bash
# Via pip (recommended)
pip install oc_py

# From source
git clone https://github.com/mshemuni/oc_py.git
cd oc_py
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1. Load Data

```python
from ocpy import OC

# Load O–C data from CSV/Excel
data = OC("my_data.csv")
```

### 2. Plot Raw Data

```python
from ocpy import Plot

Plot.plot_data(data)
```

Supports labels, error bars, and custom plotting options:

```python
Plot.plot_data(data, x_col="cycle", y_col="oc", plot_kwargs={"color": "green"})
```

---

### 3. Fit Models

#### Bayesian Fitting (PyMC)

```python
from ocpy import Linear, Quadratic, Parameter, OCPyMC, Plot

# Define model components
lin = Linear(a=Parameter(0.0), b=Parameter(0.0))
quad = Quadratic(q=Parameter(0.0))

# Fit using PyMC
model_fit = OCPyMC(data)
model_fit.fit([lin, quad], draws=2000, tune=1000, chains=4)

# Plot data + model + residuals
Plot.plot(data, model=model_fit, res=True)
```

Includes posterior median, uncertainty bands, and residuals.

---

#### Frequentist Fitting (LMFit)

```python
from ocpy import OCLMFit

# Fit using LMFit result object
lmfit_result = OCLMFit(data)
Plot.plot(data, model=lmfit_result, res=True)
```

---

#### Manual Model Components

```python
from ocpy import Sinusoidal, Plot

components = [
    Linear(a=0.01, b=0.0),
    Sinusoidal(amp=0.02, P=365.25)
]

Plot.plot(data, model=components, res=True)
```

---

### 4. Posterior Analysis

#### Corner Plots

```python
# Using corner library
fig = Plot.plot_corner(model_fit, cornerstyle="corner")

# Using ArviZ
fig = Plot.plot_corner(model_fit, cornerstyle="arviz")
```

#### Trace Plots

```python
axes = Plot.plot_trace(model_fit)
```

---

## Project Structure

* `src/ocpy/data.py` — Data container and O–C arithmetic.
* `src/ocpy/oc.py` — Model definitions and base classes.
* `src/ocpy/oc_pymc.py` — Bayesian fitting engine.
* `src/ocpy/oc_lmfit.py` — LMFit frequentist fitting engine.
* `docs/exe/` — Examples using real datasets (e.g., NY Vir, DD CrB).

---

## Contributing

Contributions are welcome! Please:

1. Follow `flake8` and `mypy` style checks.
2. Add tests for new components in `tests/`.
3. Submit PRs with descriptive messages.

---

## Support

Found a bug or need help? [Open an issue](https://github.com/mshemuni/oc_py/issues) with:

* Problem description
* Minimal reproducible example
* Environment details (Python version, OS)

---

## License

GPL-3.0 License — See `LICENSE` file for details.

