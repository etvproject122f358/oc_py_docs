from __future__ import annotations
from typing import List

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt

from .oc import OC, Linear, Quadratic, Keplerian, Sinusoidal, Parameter, ModelComponent


class OCPyMC(OC):
    """
    Bayesian O−C analysis using PyMC.

    This class provides probabilistic modeling and inference for
    O−C (Observed minus Calculated) data using Markov Chain Monte Carlo
    (MCMC) sampling via the PyMC library.

    Model components (e.g. linear, quadratic, sinusoidal, Keplerian)
    are combined additively to construct the expected O−C signal,
    while parameter uncertainties and correlations are fully
    propagated through the posterior distribution.

    Notes
    -----
    The probabilistic model assumes:

    .. math::

        (O - C)_i \\sim \\mathcal{N}(\\mu_i, \\sigma_i)

    where :math:`\\mu_i` is the sum of model component contributions
    and :math:`\\sigma_i` is taken from the observational uncertainties.

    This class is intended for full Bayesian inference, in contrast
    to :class:`OCLMFit`, which performs deterministic least-squares
    optimization.
    """
    math = pm.math

    def _to_param(self, x, *, default: float = 0.0, min_: float | None = None, max_: float | None = None,
                  fixed: bool = False, std: float | None = None) -> Parameter:
        """
        Convert input into a :class:`Parameter` instance.

        If ``x`` is already a :class:`Parameter`, it is returned unchanged.
        If ``x`` is a numeric value, it is used as the initial value.
        If ``x`` is ``None``, the provided ``default`` is used.

        Parameters
        ----------
        x
            Numeric value, ``Parameter`` instance, or ``None``.
        default
            Default value used when ``x`` is ``None``.
        min_
            Lower bound for the parameter.
        max_
            Upper bound for the parameter.
        fixed
            Whether the parameter is fixed (not sampled).
        std
            Prior standard deviation.

        Returns
        -------
        Parameter
            A fully initialized parameter object.
        """
        self.logger.debug("Converting input to Parameter")

        if isinstance(x, Parameter):
            return x
        return Parameter(value=default if x is None else x, min=min_, max=max_, fixed=fixed, std=std)

    def fit(
            self,
            model_components: List[ModelComponent],
            *,
            draws: int = 2000,
            tune: int = 2000,
            chains: int = 4,
            target_accept: float = 0.9,
            random_seed: int | None = None,
            progressbar: bool = True,
            return_model: bool = False,
            **kwargs
    ) -> az.InferenceData | pm.Model:
        """
        Perform Bayesian inference for an O−C model using MCMC sampling.

        Parameters
        ----------
        model_components
            List of model component instances to be combined additively.
        draws
            Number of posterior samples per chain.
        tune
            Number of tuning (burn-in) steps.
        chains
            Number of MCMC chains.
        target_accept
            Target acceptance probability for the NUTS sampler.
        random_seed
            Random seed for reproducibility.
        progressbar
            Whether to display the sampling progress bar.
        return_model
            If True, return the constructed PyMC model without sampling.
        **kwargs
            Additional keyword arguments passed to ``pm.sample``.

        Returns
        -------
        arviz.InferenceData or pymc.Model
            Posterior samples or the PyMC model if ``return_model=True``.

        Notes
        -----
        Each free parameter is assigned a Normal or Truncated Normal prior,
        depending on whether bounds are specified. Fixed parameters are
        treated as deterministic constants.

        The total model is constructed as:

        .. math::

            \\mu(E) = \\sum_k f_k(E; \\theta_k)

        where each :math:`f_k` is a model component.
        """
        self.logger.info("Building Bayesian O−C model with PyMC")

        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)
        sigma_i = np.asarray(self.data["minimum_time_error"].to_numpy(), dtype=float)

        if np.isnan(sigma_i).any():
            self.logger.error("Found NaN in 'minimum_time_error'.")
            raise ValueError("Found NaN in 'minimum_time_error'.")

        for c in model_components:
            if hasattr(c, "set_math"):
                c.set_math(self.math)

        def _rv(name: str, par: Parameter):
            # Create a PyMC random variable (or deterministic) from a Parameter

            val = float(getattr(par, "value", 0.0) or 0.0)
            sd = getattr(par, "std", None)
            lo = getattr(par, "min", None)
            hi = getattr(par, "max", None)
            fix = bool(getattr(par, "fixed", False))

            if fix:
                return pm.Deterministic(name, pt.as_tensor_variable(val))

            if sd is None or sd <= 0:
                sd = 1.0

            if (lo is not None and np.isfinite(lo)) or (hi is not None and np.isfinite(hi)):
                lower = float(lo) if lo is not None else None
                upper = float(hi) if hi is not None else None
                return pm.TruncatedNormal(name, mu=val, sigma=float(sd), lower=lower, upper=upper, initval=val)

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

            if return_model:
                return model

            if "cores" not in kwargs:
                kwargs["cores"] = min(chains, 4)

            if "init" not in kwargs:
                kwargs["init"] = "adapt_diag"

            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=progressbar,
                **kwargs
            )

        return idata

    def residue(self, idata: az.InferenceData, *, x_col: str = "cycle", y_col: str = "oc") -> "OCPyMC":
        """
        Compute residual O−C values using the posterior median model.

        Parameters
        ----------
        idata
            InferenceData object returned by ``fit``.
        x_col
            Column name of the independent variable.
        y_col
            Column name of observed O−C values.

        Returns
        -------
        OCPyMC
            New instance containing residuals computed as
            observed minus posterior median model.
        """
        self.logger.info("LOGGONG 3")

        y_model = idata.posterior["y_model"]
        yfit = y_model.median(dim=("chain", "draw")).values

        return OCPyMC(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list() if "minimum_time_error" in self.data else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(self.data[y_col].to_numpy(dtype=float) - yfit).tolist() if y_col in self.data else None,
        )

    def fit_linear(self, *, a: float | Parameter | None = None, b: float | Parameter | None = None, **kwargs):
        """
        Fit a linear O−C model using Bayesian inference.

        Parameters
        ----------
        a
            Linear slope parameter.
        b
            Constant offset parameter.
        **kwargs
            Passed directly to :meth:`fit`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples of the fitted model.
        """
        self.logger.debug("Preparing linear Bayesian O−C fit")

        lin = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([lin], **kwargs)

    def fit_quadratic(self, *, q: float | Parameter | None = None, **kwargs) -> az.InferenceData:
        """
        Fit a quadratic O−C model using Bayesian inference.

        Parameters
        ----------
        q
            Quadratic coefficient.
        **kwargs
            Passed directly to :meth:`fit`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples of the fitted model.
        """
        self.logger.info("Fitting Quadratic O−C model")

        comp = Quadratic(q=self._to_param(q, default=0.0))
        return self.fit([comp], **kwargs)

    def fit_sinusoidal(self, *, amp: float | Parameter | None = None, P: float | Parameter | None = None,
                       **kwargs) -> az.InferenceData:
        """
        Fit a sinusoidal O−C model using Bayesian inference.

        This model is useful for detecting periodic timing variations,
        such as those caused by a third body or cyclic processes.

        Parameters
        ----------
        amp
            Amplitude of the sinusoidal signal.
        P
            Period of the signal (in cycles).
        **kwargs
            Passed directly to :meth:`fit`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples of the fitted model.
        """
        self.logger.info("Fitting sinusoidal O−C model")

        comp = Sinusoidal(amp=self._to_param(amp, default=1e-3), P=self._to_param(P, default=1000.0))
        return self.fit([comp], **kwargs)

    def fit_keplerian(self, *, amp: float | Parameter | None = None, e: float | Parameter | None = None,
                      omega: float | Parameter | None = None, P: float | Parameter | None = None,
                      T0: float | Parameter | None = None, name: str | None = None, **kwargs) -> az.InferenceData:
        """
        Fit a Keplerian O−C model using Bayesian inference.

        This model represents the light-time effect (LITE) caused
        by an orbiting third body.

        Parameters
        ----------
        amp
            Semi-amplitude of the timing signal.
        e
            Orbital eccentricity.
        omega
            Argument of periastron (degrees).
        P
            Orbital period.
        T0
            Time of periastron passage.
        name
            Optional name for the Keplerian component.
        **kwargs
            Passed directly to :meth:`fit`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples of the fitted model.
        """
        comp = Keplerian(
            amp=self._to_param(amp, default=0.001),
            e=self._to_param(e, default=0.1),
            omega=self._to_param(omega, default=90.0),
            P=self._to_param(P, default=1000.0),
            T0=self._to_param(T0, default=0.0),
            name=name or "keplerian1",
        )
        self.logger.info("Fitting keplerian O−C model")

        return self.fit([comp], **kwargs)

    def fit_lite(self, **kwargs) -> az.InferenceData:
        """
        Fit a light-time effect (LITE) model.

        This is a convenience wrapper around :meth:`fit_keplerian`
        using default Keplerian parameterization.

        Parameters
        ----------
        **kwargs
            Passed directly to :meth:`fit_keplerian`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples of the fitted model.
        """
        self.logger.info("Fitting lite O−C model")

        return self.fit_keplerian(**kwargs)

    def fit_parabola(self, *, q: float | Parameter | None = None, a: float | Parameter | None = None,
                     b: float | Parameter | None = None, **kwargs) -> az.InferenceData:
        """
        Fit a combined quadratic + linear O−C model.

        This model represents secular period change (quadratic term)
        combined with a linear ephemeris correction.

        Parameters
        ----------
        q
            Quadratic coefficient.
        a
            Linear slope parameter.
        b
            Constant offset parameter.
        **kwargs
            Passed directly to :meth:`fit`.

        Returns
        -------
        arviz.InferenceData
            Posterior samples of the fitted model.
        """

        self.logger.info("Fitting parabola O−C model")

        quad = Quadratic(q=self._to_param(q, default=0.0))
        lin = Linear(a=self._to_param(a, default=0.0), b=self._to_param(b, default=0.0))
        return self.fit([quad, lin], **kwargs)
