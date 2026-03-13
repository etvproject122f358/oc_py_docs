from __future__ import annotations
from typing import Dict, List, Optional, Literal, Self, Union
import warnings

import matplotlib
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from matplotlib import pyplot as plt

from .oc import OC, Linear, Quadratic, Keplerian, Sinusoidal, Parameter, ModelComponent
from .visualization import Plot


class OCPyMC(OC):
    """
    A probabilistic O–C (Observed minus Calculated) modeling class using PyMC.

    This class extends the `OC` base class to provide Bayesian inference
    for O–C data, allowing modeling with various components such as linear,
    quadratic, sinusoidal, and Keplerian functions. It leverages PyMC and
    ArviZ for probabilistic sampling, posterior analysis, and visualization.

    Parameters
    ----------
    Inherited from OC
        Accepts the same initialization parameters as `OC`, such as cycle
        counts, O–C values, errors, weights, labels, and minimum types.

    Attributes
    ----------
    math : module
        Set to `pymc.math` to allow model components to use PyMC tensor operations.

    Methods
    -------
    fit(model_components, *, draws, tune, chains, cores, target_accept, random_seed, progressbar, return_model, **kwargs)
        Fit the provided model components to the O–C data using Bayesian sampling.
        Returns an `InferenceData` object with posterior samples.

    clean(inference_data, drop_chains=0, filter_outliers=True, iqr_multiplier=4.0)
        Post-process `InferenceData` to drop chains or filter extreme outliers.

    residue(inference_data, *, x_col="cycle", y_col="oc")
        Return a new `OCPyMC` instance containing residuals between observed
        and model-predicted O–C values.

    fit_linear(*, a=None, b=None, cores=None, **kwargs)
        Fit a linear component (a * x + b) to the data.

    fit_quadratic(*, q=None, cores=None, **kwargs)
        Fit a quadratic component (q * x^2) to the data.

    fit_parabola(*, q=None, a=None, b=None, cores=None, **kwargs)
        Fit a combination of quadratic and linear components to the data.

    fit_sinusoidal(*, amp=None, P=None, cores=None, **kwargs)
        Fit a sinusoidal component (amp * sin(2π x / P)) to the data.

    fit_keplerian(*, amp=None, e=None, omega=None, P=None, T0=None, name=None, cores=None, **kwargs)
        Fit a Keplerian (orbital) component to the data.

    fit_lite(**kwargs)
        Alias for `fit_keplerian`; provides a simplified interface for fitting
        a single Keplerian component.

    corner(inference_data, cornerstyle="corner", units=None, **kwargs)
        Plot a corner plot of posterior distributions using `Plot.plot_corner`.

    trace(inference_data, **kwargs)
        Plot trace plots for posterior chains using `Plot.plot_trace`.

    Notes
    -----
    - This class is designed for O–C analyses in eclipsing binaries, exoplanet
      timing, and other periodic astrophysical phenomena.
    - Supports both deterministic and probabilistic parameters, including
      truncated distributions for constrained parameters.
    - Works seamlessly with multiple model components, automatically managing
      parameter naming and aggregation.
    """
    math = pm.math

    def _to_param(self, x, *, default: float = 0.0, min_: float | None = None, max_: float | None = None,
                  fixed: bool = False, std: float | None = None) -> Parameter:
        """
        Convert a value or existing Parameter into a `Parameter` object.

        This method ensures that any input intended as a model parameter
        is wrapped as a `Parameter` instance, setting default values,
        bounds, fixed status, and standard deviation as needed.

        Parameters
        ----------
        x : float or Parameter or None
            The input value to convert. If `x` is already a `Parameter`, it
            is returned unchanged. If `None`, `default` is used.
        default : float, optional
            The default value to use if `x` is `None` (default is 0.0).
        min_ : float or None, optional
            Optional lower bound for the parameter (default is None).
        max_ : float or None, optional
            Optional upper bound for the parameter (default is None).
        fixed : bool, optional
            Whether the parameter should be treated as fixed (not sampled)
            in Bayesian inference (default is False).
        std : float or None, optional
            Standard deviation for the parameter (used when defining priors)
            if applicable (default is None).

        Returns
        -------
        Parameter
            A `Parameter` object encapsulating value, bounds, fixed status, and std.

        Notes
        -----
        - This helper is mainly used internally to standardize parameter inputs
          before fitting model components with PyMC.
        """
        if isinstance(x, Parameter):
            return x
        return Parameter(value=default if x is None else x, min=min_, max=max_, fixed=fixed, std=std)

    def fit(self,
            model_components: List[ModelComponent],
            *,
            draws: int = 2000,
            tune: int = 2000,
            chains: int = 4,
            cores: Optional[int] = None,
            target_accept: Optional[float] = None,
            random_seed: Optional[int] = None,
            progressbar: bool = True,
            return_model: bool = False,
            **kwargs
            ) -> az.InferenceData | pm.Model:
        """
        Fit a model to the observed O–C (observed minus calculated) data using Bayesian inference.

        This method constructs a PyMC probabilistic model using the specified
        `model_components`, applies priors based on their parameters, and samples
        from the posterior distribution.

        Parameters
        ----------
        model_components : list of ModelComponent
            List of model components to fit (e.g., Linear, Quadratic, Sinusoidal, Keplerian).
            Each component must define a `model_func` and optional `params`.
        draws : int, optional
            Number of posterior samples to draw after tuning (default: 2000).
        tune : int, optional
            Number of tuning (burn-in) steps (default: 2000).
        chains : int, optional
            Number of Markov chains to run (default: 4).
        cores : int or None, optional
            Number of CPU cores to use. Defaults to number of chains if None.
        target_accept : float or None, optional
            Target acceptance probability for NUTS sampler (default: None).
        random_seed : int or None, optional
            Random seed for reproducibility.
        progressbar : bool, optional
            Whether to show the sampling progress bar (default: True).
        return_model : bool, optional
            If True, return the PyMC `Model` object without sampling.
        **kwargs
            Additional keyword arguments passed to `pm.sample()` or sampler step methods.

        Returns
        -------
        arviz.InferenceData or pm.Model
            If `return_model` is False, returns an ArviZ `InferenceData` object
            containing posterior samples, posterior predictive samples, and deterministic
            variables for model evaluation. If `return_model` is True, returns the
            PyMC `Model` object without sampling.

        Notes
        -----
        - All model components’ parameters are converted to `Parameter` objects using `_to_param`.
        - Observational uncertainties are taken from `minimum_time_error`; NaNs will raise a ValueError.
        - Deterministic variables `y_model` and, if possible, `y_model_dense` are created for visualization.
        - Components marked as `_expensive` bypass dense evaluation to avoid heavy computations.
        """

        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)
        sigma_i = np.asarray(self.data["minimum_time_error"].to_numpy(), dtype=float)

        if np.isnan(sigma_i).any():
            raise ValueError("Found NaN in 'minimum_time_error'.")

        for c in model_components:
            if hasattr(c, "set_math"):
                c.set_math(self.math)

        def _rv(name: str, par: Parameter):
            val = float(getattr(par, "value", 0.0) or 0.0)
            sd = getattr(par, "std", None)
            lo = getattr(par, "min", None)
            hi = getattr(par, "max", None)
            fix = bool(getattr(par, "fixed", False))

            if fix:
                return pm.Deterministic(name, pt.as_tensor_variable(val))

            if sd is None or sd <= 0:
                sd = max(abs(val) * 0.1, 1e-6)

            if (lo is not None and np.isfinite(lo)) or (hi is not None and np.isfinite(hi)):
                lower = float(lo) if lo is not None else None
                upper = float(hi) if hi is not None else None

                safe_val = val
                eps = 1e-5
                if lower is not None and safe_val <= lower:
                    safe_val = lower + eps
                if upper is not None and safe_val >= upper:
                    safe_val = upper - eps
                if lower is not None and safe_val < lower:
                    safe_val = lower
                if upper is not None and safe_val > upper:
                    safe_val = upper

                return pm.TruncatedNormal(name, mu=val, sigma=float(sd), lower=lower, upper=upper, initval=safe_val)

            return pm.Normal(name, mu=val, sigma=float(sd), initval=val)

        with pm.Model() as model:
            base_names = [getattr(c, 'name', c.__class__.__name__.lower()) for c in model_components]
            counts = {name: base_names.count(name) for name in base_names}
            seen = {name: 0 for name in base_names}

            prefixes = []
            for name in base_names:
                seen[name] += 1
                if counts[name] > 1:
                    prefixes.append(f"{name}{seen[name]}_")
                else:
                    prefixes.append(f"{name}_")
            comp_rvs = {}

            for comp, pref in zip(model_components, prefixes):
                rvs = {}
                for pname, par in getattr(comp, "params", {}).items():
                    rvs[pname] = _rv(pref + pname, par)
                comp_rvs[pref] = rvs

            mus = []
            for comp, pref in zip(model_components, prefixes):
                mus.append(comp.model_func(x, **comp_rvs[pref]))

            mu_total = mus[0] if len(mus) == 1 else sum(mus)

            pm.Deterministic("y_model", mu_total)
            pm.Normal("y_obs", mu=mu_total, sigma=sigma_i, observed=y)

            has_expensive = any(getattr(c, "_expensive", False) for c in model_components)

            if not has_expensive:
                xmin, xmax = np.min(x), np.max(x)
                margin = (xmax - xmin) * 0.05
                dense_x_vals = np.linspace(xmin - margin, xmax + margin, 500)

                mus_dense = []
                for comp, pref in zip(model_components, prefixes):
                    mus_dense.append(comp.model_func(dense_x_vals, **comp_rvs[pref]))

                mu_total_dense = mus_dense[0] if len(mus_dense) == 1 else sum(mus_dense)
                pm.Deterministic("y_model_dense", mu_total_dense)
                pm.Deterministic("dense_x", pt.as_tensor_variable(dense_x_vals))

            if return_model:
                return model

            # Pack sample arguments
            sample_kwargs = kwargs.copy()

            if cores is not None:
                sample_kwargs["cores"] = min(cores, chains)
            elif "cores" not in sample_kwargs:
                sample_kwargs["cores"] = chains

            if target_accept is not None:
                sample_kwargs["target_accept"] = target_accept

            if has_expensive and "step" not in sample_kwargs:
                sample_kwargs["step"] = pm.DEMetropolisZ()

            if "step" in sample_kwargs and callable(sample_kwargs["step"]) and not isinstance(sample_kwargs["step"],
                                                                                              pm.step_methods.arraystep.ArrayStep):
                sample_kwargs["step"] = sample_kwargs["step"]()

            inference_data = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=progressbar,
                **sample_kwargs
            )

        inference_data.attrs["_model_components"] = model_components
        inference_data.attrs["_model_prefixes"] = prefixes

        return inference_data

    def clean(self,
              inference_data: az.InferenceData,
              drop_chains: int = 0,
              filter_outliers: bool = True,
              iqr_multiplier: float = 4.0
              ) -> az.InferenceData:
        """
        Clean an ArviZ InferenceData object by optionally removing chains and filtering outliers.

        This method allows post-processing of MCMC samples to improve data quality for
        visualization or further analysis. Chains with extreme deviations can be dropped,
        and outliers in the posterior distributions can be filtered using the interquartile
        range (IQR) method.

        Parameters
        ----------
        inference_data : arviz.InferenceData
            The posterior samples to clean, typically returned by `OCPyMC.fit()`.
        drop_chains : int, optional
            Number of chains to drop based on deviation from the median across chains
            (default: 0, i.e., keep all chains).
        filter_outliers : bool, optional
            Whether to remove outliers from the posterior samples (default: True).
        iqr_multiplier : float, optional
            Multiplier for the interquartile range (IQR) to define outlier bounds
            (default: 4.0). Samples outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
            are removed.

        Returns
        -------
        arviz.InferenceData
            A new `InferenceData` object with cleaned posterior samples, while preserving
            deterministic variables and model component metadata (`_model_components` and
            `_model_prefixes`).

        Notes
        -----
        - Outlier filtering is applied independently to each parameter in the posterior
          (excluding deterministic variables like `y_model`, `y_model_dense`, `y_obs`, `dense_x`).
        - Chains are ranked by their deviation from the overall median, and the `drop_chains`
          largest deviations are removed.
        - The cleaned dataset preserves the original structure of `InferenceData`, including
          `posterior`, `prior`, and `observed_data` groups.
        - Any errors encountered during filtering are issued as warnings without stopping execution.
        """
        posterior_data = inference_data.posterior
        chain_coords = posterior_data.coords["chain"].values
        chains_to_keep = list(chain_coords)

        if drop_chains > 0:
            var_names = [var_name for var_name in posterior_data.data_vars if
                         getattr(posterior_data[var_name], "ndim", 0) == 2 and var_name not in {"y_model",
                                                                                                "y_model_dense",
                                                                                                "y_obs", "dense_x"}]
            if drop_chains >= len(chain_coords):
                raise ValueError("drop_chains must be less than the total number of chains.")

            chain_distances = []
            for chain in chain_coords:
                distance = 0.0
                for var_name in var_names:
                    overall_median = float(posterior_data[var_name].median())
                    chain_median = float(posterior_data[var_name].sel(chain=chain).median())
                    std = float(posterior_data[var_name].std())
                    if std > 1e-10:
                        distance += abs(chain_median - overall_median) / std
                chain_distances.append((chain, distance))

            chain_distances.sort(key=lambda x: x[1], reverse=True)
            chains_to_drop = [chain for chain, dist_val in chain_distances[:drop_chains]]
            chains_to_keep = [chain for chain in chain_coords if chain not in chains_to_drop]
            posterior_sub = posterior_data.sel(chain=chains_to_keep)
        else:
            posterior_sub = posterior_data

        mask = None
        if filter_outliers:
            var_names = [var_name for var_name in posterior_sub.data_vars if
                         getattr(posterior_sub[var_name], "ndim", 0) == 2 and var_name not in {"y_model",
                                                                                               "y_model_dense", "y_obs",
                                                                                               "dense_x"}]
            stacked = posterior_sub.stack(sample=("chain", "draw"))
            mask = np.ones(stacked.sizes["sample"], dtype=bool)

            for var_name in var_names:
                values_array = stacked[var_name].values
                quartile_1 = np.percentile(values_array, 25)
                quartile_3 = np.percentile(values_array, 75)
                interquartile_range = quartile_3 - quartile_1
                if interquartile_range > 1e-10:
                    lower_bound = quartile_1 - iqr_multiplier * interquartile_range
                    upper_bound = quartile_3 + iqr_multiplier * interquartile_range
                    mask = mask & (values_array >= lower_bound) & (values_array <= upper_bound)

        new_groups = {}
        for group_name in inference_data._groups:
            group_dataset = getattr(inference_data, group_name)
            if "chain" in group_dataset.dims and "draw" in group_dataset.dims:
                if drop_chains > 0:
                    try:
                        group_dataset = group_dataset.sel(chain=chains_to_keep)
                    except KeyError:
                        pass

                if filter_outliers and mask is not None:
                    try:
                        stacked = group_dataset.stack(sample=("chain", "draw"))
                        filtered = stacked.isel(sample=mask)
                        n_draws = filtered.sizes["sample"]

                        # Prepare the dataset for dimensionality reshaping cleanly
                        filtered = filtered.drop_vars(["chain", "draw", "sample"], errors="ignore")
                        filtered = filtered.rename({"sample": "draw"})
                        filtered = filtered.assign_coords({"draw": np.arange(n_draws)})

                        # Re-add chain dimension
                        group_dataset = filtered.expand_dims({"chain": [0]}).transpose("chain", "draw", ...)
                    except Exception as error_msg:
                        warnings.warn(f"clean() encountered an issue on {group_name}: {error_msg}")

            new_groups[group_name] = group_dataset

        cleaned = az.InferenceData(**new_groups)
        for attr_key in ("_model_components", "_model_prefixes"):
            if attr_key in getattr(inference_data, "attrs", {}):
                cleaned.attrs[attr_key] = inference_data.attrs[attr_key]
        return cleaned

    def residue(self, inference_data: az.InferenceData, *, x_col: str = "cycle", y_col: str = "oc") -> Self:
        """
        Compute the residuals between observed O–C values and the model predictions.

        This method generates a new `OCPyMC` instance where the O–C values are replaced
        by the difference between the observed values and the median model predictions
        from a fitted PyMC InferenceData object.

        Parameters
        ----------
        inference_data : arviz.InferenceData
            The posterior samples returned by `OCPyMC.fit()` containing the fitted model.
        x_col : str, optional
            Name of the column in `self.data` containing the independent variable (cycle),
            used to evaluate the model (default: "cycle").
        y_col : str, optional
            Name of the column in `self.data` containing the observed O–C values
            (default: "oc").

        Returns
        -------
        OCPyMC
            A new `OCPyMC` instance with the same metadata as the original, but with the
            `oc` values replaced by the residuals (observed minus fitted).

        Notes
        -----
        - The model predictions are computed as the median across all chains and draws in
          the posterior (`inference_data.posterior["y_model"]`).
        - Only the `oc` values are updated; all other attributes (weights, labels, minimum times, etc.) are preserved.
        - This method is useful for examining the remaining structure or noise after fitting a model.
        """
        y_model = inference_data.posterior["y_model"]
        y_fit = y_model.median(dim=("chain", "draw")).values

        return OCPyMC(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list() if "minimum_time_error" in self.data else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(self.data[y_col].to_numpy(dtype=float) - y_fit).tolist() if y_col in self.data else None,
        )

    def fit_linear(self, *, a: float | Parameter | None = None, b: float | Parameter | None = None,
                   cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        """
        Fit a linear O–C model (O–C = a * cycle + b) using PyMC Bayesian inference.

        The linear model is represented as:
            O–C = a * x + b
        where `x` is the cycle number.

        Parameters
        ----------
        a : float or Parameter, optional
            Initial value or `Parameter` object for the slope of the line.
            If `None`, defaults to 0.0.
        b : float or Parameter, optional
            Initial value or `Parameter` object for the intercept of the line.
            If `None`, defaults to 0.0.
        cores : int, optional
            Number of CPU cores to use for sampling. If `None`, the number of cores
            defaults to the number of chains.
        **kwargs
            Additional keyword arguments passed to `OCPyMC.fit()`, such as `draws`,
            `tune`, `chains`, `target_accept`, etc.

        Returns
        -------
        arviz.InferenceData
            Posterior samples for the linear model parameters and deterministic
            predictions for the O–C values.

        Notes
        -----
        - The slope (`a`) and intercept (`b`) are treated as PyMC random variables
          unless provided as fixed `Parameter` objects with `fixed=True`.
        - Use `residue()` on the returned model to compute residuals after fitting.
        - This method is a convenience wrapper that internally creates a `Linear`
          component and calls the general `fit()` method.
        """
        lin = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([lin], cores=cores, **kwargs)

    def fit_quadratic(self, *, q: float | Parameter | None = None, cores: Optional[int] = None,
                      **kwargs) -> az.InferenceData:
        """
        Fit a quadratic O–C model (O–C = q * cycle^2) using PyMC Bayesian inference.

        The quadratic model is represented as:
            O–C = q * x^2
        where `x` is the cycle number.

        Parameters
        ----------
        q : float or Parameter, optional
            Initial value or `Parameter` object for the quadratic coefficient.
            If `None`, defaults to 0.0.
        cores : int, optional
            Number of CPU cores to use for sampling. If `None`, the number of cores
            defaults to the number of chains.
        **kwargs
            Additional keyword arguments passed to `OCPyMC.fit()`, such as `draws`,
            `tune`, `chains`, `target_accept`, etc.

        Returns
        -------
        arviz.InferenceData
            Posterior samples for the quadratic coefficient and deterministic
            predictions for the O–C values.

        Notes
        -----
        - The coefficient `q` is treated as a PyMC random variable unless provided
          as a fixed `Parameter` with `fixed=True`.
        - Use `residue()` on the returned model to compute residuals after fitting.
        - This method is a convenience wrapper that internally creates a `Quadratic`
          component and calls the general `fit()` method.
        """
        comp = Quadratic(q=self._to_param(q, default=0.0))
        return self.fit([comp], cores=cores, **kwargs)

    def fit_sinusoidal(self, *, amp: float | Parameter | None = None, P: float | Parameter | None = None,
                       cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        """
        Fit a sinusoidal O–C model (O–C = amp * sin(2π * cycle / P)) using PyMC Bayesian inference.

        The sinusoidal model is represented as:
            O–C = amp * sin(2π * x / P)
        where `x` is the cycle number.

        Parameters
        ----------
        amp : float or Parameter, optional
            Initial value or `Parameter` object for the sinusoidal amplitude.
            If `None`, defaults to 1e-3.
        P : float or Parameter, optional
            Initial value or `Parameter` object for the period of the sinusoid.
            If `None`, defaults to 1000.0.
        cores : int, optional
            Number of CPU cores to use for sampling. If `None`, the number of cores
            defaults to the number of chains.
        **kwargs
            Additional keyword arguments passed to `OCPyMC.fit()`, such as `draws`,
            `tune`, `chains`, `target_accept`, etc.

        Returns
        -------
        arviz.InferenceData
            Posterior samples for the sinusoidal parameters and deterministic
            predictions for the O–C values.

        Notes
        -----
        - The parameters `amp` and `P` are treated as PyMC random variables unless
          provided as fixed `Parameter` objects with `fixed=True`.
        - Use `residue()` on the returned model to compute residuals after fitting.
        - This method is a convenience wrapper that internally creates a `Sinusoidal`
          component and calls the general `fit()` method.
        """
        comp = Sinusoidal(amp=self._to_param(amp, default=1e-3), P=self._to_param(P, default=1000.0))
        return self.fit([comp], cores=cores, **kwargs)

    def fit_keplerian(self, *, amp: float | Parameter | None = None, e: float | Parameter | None = None,
                      omega: float | Parameter | None = None, P: float | Parameter | None = None,
                      T0: float | Parameter | None = None, name: Optional[str] = None, cores: Optional[int] = None,
                      **kwargs) -> az.InferenceData:
        """
        Fit a Keplerian O–C model using PyMC Bayesian inference.

        The Keplerian model represents the O–C variations as caused by a
        companion in an orbit, parameterized by:
            - amp  : amplitude of the timing variation
            - e    : orbital eccentricity
            - omega: argument of periastron (degrees)
            - P    : orbital period
            - T0   : time of periastron passage

        Parameters
        ----------
        amp : float or Parameter, optional
            Initial value or `Parameter` object for the amplitude of the O–C signal.
            Defaults to 0.001 if None.
        e : float or Parameter, optional
            Initial value or `Parameter` object for orbital eccentricity.
            Defaults to 0.1 if None.
        omega : float or Parameter, optional
            Initial value or `Parameter` object for the argument of periastron (degrees).
            Defaults to 90.0 if None.
        P : float or Parameter, optional
            Initial value or `Parameter` object for the orbital period.
            Defaults to 1000.0 if None.
        T0 : float or Parameter, optional
            Initial value or `Parameter` object for the time of periastron passage.
            Defaults to 0.0 if None.
        name : str, optional
            Optional name for the Keplerian component. If None, defaults to 'keplerian1'.
        cores : int, optional
            Number of CPU cores to use for sampling. If None, defaults to the number of chains.
        **kwargs
            Additional keyword arguments passed to `OCPyMC.fit()`, e.g., `draws`, `tune`,
            `chains`, `target_accept`, etc.

        Returns
        -------
        arviz.InferenceData
            Posterior samples for all Keplerian parameters, along with deterministic
            predictions for the O–C curve.

        Notes
        -----
        - The parameters are treated as PyMC random variables unless provided as
          fixed `Parameter` objects with `fixed=True`.
        - Use `residue()` on the returned model to compute residuals after fitting.
        - This method internally creates a `Keplerian` component and calls the
          general `fit()` method for Bayesian inference.
        """
        comp = Keplerian(
            amp=self._to_param(amp, default=0.001),
            e=self._to_param(e, default=0.1),
            omega=self._to_param(omega, default=90.0),
            P=self._to_param(P, default=1000.0),
            T0=self._to_param(T0, default=0.0),
            name=name or "keplerian1",
        )
        return self.fit([comp], cores=cores, **kwargs)

    def fit_lite(self, **kwargs) -> az.InferenceData:
        """
        Fit a simplified Keplerian O–C model using PyMC Bayesian inference.

        This is a convenience wrapper around `fit_keplerian` with default
        Keplerian parameters. It is designed for quick fitting without
        specifying all orbital parameters explicitly.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed directly to `fit_keplerian`, including:
            - amp  : amplitude of the O–C signal
            - e    : orbital eccentricity
            - omega: argument of periastron (degrees)
            - P    : orbital period
            - T0   : time of periastron passage
            - cores: number of CPU cores to use
            - draws, tune, chains, target_accept, etc.

        Returns
        -------
        arviz.InferenceData
            Posterior samples for the Keplerian parameters and deterministic
            predictions of the O–C curve.

        Notes
        -----
        - This method internally calls `fit_keplerian` with default parameter values:
            amp=0.001, e=0.1, omega=90.0, P=1000.0, T0=0.0
        - Use `residue()` on the returned model to compute residuals after fitting.
        - Suitable for quick exploratory fits or as an initial guess for more complex models.
        """
        return self.fit_keplerian(**kwargs)

    def fit_parabola(self, *, q: float | Parameter | None = None, a: float | Parameter | None = None,
                     b: float | Parameter | None = None, cores: Optional[int] = None, **kwargs) -> az.InferenceData:
        """
        Fit a combined quadratic and linear O–C model using PyMC Bayesian inference.

        This method models the O–C variations as a sum of a quadratic component
        (representing secular period change) and a linear component (representing
        a constant period offset).

        Parameters
        ----------
        q : float or Parameter, optional
            Quadratic coefficient for the parabolic term. Default is 0.0.
        a : float or Parameter, optional
            Linear coefficient for the linear term. Default is 0.0.
        b : float or Parameter, optional
            Constant offset term. Default is 0.0.
        cores : int, optional
            Number of CPU cores to use for PyMC sampling. Defaults to the number of chains.
        **kwargs
            Additional keyword arguments passed to the underlying `fit` method,
            such as `draws`, `tune`, `chains`, `target_accept`, and `random_seed`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples for the quadratic and linear coefficients, along
            with deterministic predictions of the O–C curve.

        Notes
        -----
        - The quadratic component captures long-term period changes.
        - The linear component models any overall offset in the period.
        - Use the `residue()` method on the returned object to compute residuals
          of the fit.
        - Recommended as a first approach when O–C variations show a parabolic trend.
        """
        quad = Quadratic(q=self._to_param(q, default=0.0))
        lin = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([quad, lin], cores=cores, **kwargs)

    def corner(self, inference_data: az.InferenceData, cornerstyle: Literal["corner", "arviz"] = "corner",
               units: Optional[Dict[str, str]] = None, **kwargs) -> Union[plt.Figure, az.plot_pair]:
        """
        Generate a corner (pairwise posterior) plot from PyMC inference results.

        This method visualizes the posterior distributions of model parameters,
        showing marginal distributions along the diagonal and pairwise correlations
        in the off-diagonal plots. It can use either the `corner` library or ArviZ
        built-in plotting depending on the `cornerstyle` argument.

        Parameters
        ----------
        inference_data : arviz.InferenceData
            The PyMC sampling results, typically returned by `fit()`, containing posterior samples.
        cornerstyle : {"corner", "arviz"}, default "corner"
            Choose the plotting backend:
            - "corner": use the corner.py library for plotting.
            - "arviz": use ArviZ's built-in pairplot functionality.
        units : dict, optional
            A mapping from parameter names to strings describing their units.
            These units will be displayed in axis labels.
        **kwargs
            Additional keyword arguments passed to the underlying plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object for the 'corner' style.
        arviz.plot_pair axes or Figure
            The ArviZ axes object when `cornerstyle='arviz'`.

        Notes
        -----
        - Useful for visual inspection of posterior distributions and parameter correlations.
        - Recommended to use after `fit()` and optionally after `clean()` for outlier removal.
        - Units help clarify parameter scales in the plot.
        """
        return Plot.plot_corner(inference_data, cornerstyle=cornerstyle, units=units, **kwargs)

    def trace(self, inference_data: az.InferenceData, **kwargs) -> matplotlib.axes.Axes:
        """
        Generate trace plots for PyMC posterior samples.

        Trace plots visualize the sampled parameter values across chains,
        allowing assessment of convergence, mixing, and sampling behavior.
        Each subplot shows the parameter values over draws for one chain.

        Parameters
        ----------
        inference_data : arviz.InferenceData
            The PyMC sampling results, typically returned by `fit()`, containing posterior samples.
        **kwargs
            Additional keyword arguments passed to the underlying plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            Array of matplotlib axes objects containing the trace plots.

        Notes
        -----
        - Useful for evaluating MCMC convergence and diagnosing sampling issues.
        - Typically used after `fit()` and optionally after `clean()` to inspect cleaned chains.
        - Can be combined with corner plots for a complete posterior overview.
        """
        return Plot.plot_trace(inference_data, **kwargs)
