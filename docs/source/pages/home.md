# ocpy: A Modular Python Suite for Eclipse Timing Variation Analysis

ocpy is an open-source, platform-independent Python-3 library designed for high-precision modeling and analysis of Eclipse Timing Variations (ETV). Developed with a modular architecture, the software provides a robust pipeline from raw data processing to complex modeling and Bayesian inference.

By leveraging standard scientific libraries, ocpy ensures reliability and reproducibility in modeling orbital period variations, whether they are periodic or cyclic driven by light-travel time effect (LiTE), dynamical perturbations, or stellar magnetic activity; or secular driven by mechanisms such as mass transfer / loss, gravitational radiation.

## Key Features & Modules

ocpy is structured into functional modules to facilitate a flexible research workflow:

- **Data Pre-processing:**
  - **Read data:** Multi-format support (ASCII, CSV, Excel) via `pandas` and `astropy`.
  - **Time conversion:** High-precision conversion of diverse time scales to BJD-TDB.
  - **Weighting and binning data:** Statistical weighting (by instrument or uncertainty) and seasonal data folding.
  - **Outlier removal:** Automated identification and removal of discrepant data points using robust statistical methods.

- **Modeling & Fitting:**
  - **Analytical Models:** Implements linear, quadratic (orbital decay), and Keplerian (LiTE) models based on Irwin (1952) using `lmfit` and `scipy`.
  - **Dynamical N-Body Integration:** Advanced Newtonian modeling of multi-body systems using the REBOUND integrator.
  - **Stellar Activity:** Models for magnetic activity-induced variations based on Applegate (1992), Völschow (2016), and Lanza (2020).
  - **Orbital Decay & Growth:** Quadratic functions to model secular orbital period variations.
  - **User-Defined Models:** A flexible interface for integrating custom analytical ETV functions combining different effects.

- **Statistical Analysis:**
  - **MCMC Sampling:** Full Bayesian parameter estimation and posterior probability mapping using PyMC.
  - **Model Comparison:** Statistical model selection (AIC, BIC, $\chi^2_\nu$) to distinguish between competing models.

- **Visualization & Output:**
  - **Plotting:** Graphics and plots for publication-ready O-C curves, and corner plots (parameter correlations).
  - **Output parameters:** Automated generation of formatted tables (LaTeX, CSV) for direct inclusion in scientific papers.
