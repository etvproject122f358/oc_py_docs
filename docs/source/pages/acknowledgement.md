# Acknowledgement

We gratefully acknowledge the support by The Scientific and Technological Research Council of Turkey (TÜBİTAK) with the project 122F358.

## Scientific Heritage

This software implements and builds upon several key theoretical frameworks for modeling Observed‑minus‑Calculated (O–C) variations in close binary systems.

**Light‑Time Effect (LiTE):**
For Keplerian orbital delay, the code utilizes the analytical formulations provided by [Esmer 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...648A..85E/abstract).

**Newtonian Models:**
Modeling of Newtonian (N‑body) interactions is based on the integration procedures detailed in [Esmer 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.6050E/abstract), utilizing the REBOUND N‑body code.

**Stellar Activity & Magnetic Effects:**
Period variations resulting from magnetic activity cycles are implemented following:

* The [Applegate 1992](https://ui.adsabs.harvard.edu/abs/1992ApJ...385..621A/abstract) mechanism regarding quadrupole moment variations.
* The dynamo‑theory‑based models refined by [Völschow 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...620A..42V/abstract).

## Software & Libraries

The development of this code relies heavily on the open‑source Python ecosystem. We particularly acknowledge the use of the following tools:


* **arviz** — for Bayesian posterior diagnostics ([GitHub](https://github.com/arviz-devs/arviz))
* **corner** — for visualizing multidimensional samples ([GitHub](https://github.com/dfm/corner.py))
* **lmfit** — for nonlinear least‑squares optimization ([GitHub](https://github.com/lmfit/lmfit-py))
* **matplotlib** — for scientific plotting ([GitHub](https://github.com/matplotlib/matplotlib))
* **NumPy** — for numerical computing and optimization ([GitHub](https://github.com/numpy/numpy))
* **pandas** — for data structures and time series analysis ([GitHub](https://github.com/pandas-dev/pandas))
* **PyMC** — for Bayesian inference and MCMC sampling ([GitHub](https://github.com/pymc-devs/pymc))
* **REBOUND** — for high‑accuracy N‑body integrations ([GitHub](https://github.com/hannorein/rebound))
* **SciPy** — for scientific computing and optimization ([GitHub](https://github.com/scipy/scipy))


