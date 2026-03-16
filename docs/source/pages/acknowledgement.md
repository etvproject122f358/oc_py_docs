# Acknowledgement

We gratefully acknowledge the support by The Scientific and Technological Research Council of Turkey (TÜBİTAK) with the project 122F358.

## Scientific Heritage

This software implements and builds upon several key theoretical frameworks for modeling Observed‑minus‑Calculated (O–C) variations in close binary systems.

**Light‑Time Effect (LiTE):**
For Keplerian orbital delay, the code utilizes the analytical formulations provided by Irwin (1952; [DOI: 10.1086/145604](https://doi.org/10.1086/145604)).

**Dynamical Effects:**
Modeling of Newtonian (N‑body) interactions is based on the integration procedures detailed in Esmer et al. (2022; “Detection of two additional circumbinary planets around *Kepler‑451*”; MNRAS 511, 5207–5216; [DOI: 10.1093/mnras/stac357](https://doi.org/10.1093/mnras/stac357)) and Esmer et al. (2023; “Testing the planetary hypothesis of *NY Virginis*…”, MNRAS 525, 6050–6063; [DOI: 10.1093/mnras/stad2648](https://doi.org/10.1093/mnras/stad2648)), utilizing the REBOUND N‑body code (Rein 2019; Rein & Liu 2012; [DOI: 10.1051/0004‑6361/201118085](https://doi.org/10.1051/0004-6361/201118085)).

**Stellar Activity & Magnetic Effects:**
Period variations resulting from magnetic activity cycles are implemented following:

* The Applegate (1992; [DOI: 10.1086/170967](https://doi.org/10.1086/170967)) mechanism regarding quadrupole moment variations.
* The dynamo‑theory‑based models refined by Völschow et al. (2018; [DOI: 10.1051/0004‑6361/201833506](https://doi.org/10.1051/0004-6361/201833506)).

## Software & Libraries

The development of this code relies heavily on the open‑source Python ecosystem. We particularly acknowledge the use of the following tools:

* **PyMC** — for Bayesian inference and MCMC sampling ([GitHub](https://github.com/pymc-devs/pymc))
* **REBOUND** — for high‑accuracy N‑body integrations ([GitHub](https://github.com/hannorein/rebound))
* **NumPy** — for numerical computing and optimization ([GitHub](https://github.com/numpy/numpy))
* **SciPy** — for scientific computing and optimization ([GitHub](https://github.com/scipy/scipy))
* **lmfit** — for nonlinear least‑squares optimization ([GitHub](https://github.com/lmfit/lmfit-py))
* **pandas** — for data structures and time series analysis ([GitHub](https://github.com/pandas-dev/pandas))
* **arviz** — for Bayesian posterior diagnostics ([GitHub](https://github.com/arviz-devs/arviz))
* **matplotlib** — for scientific plotting ([GitHub](https://github.com/matplotlib/matplotlib))

