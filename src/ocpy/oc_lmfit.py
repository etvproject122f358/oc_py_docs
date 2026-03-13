from typing import Union, Self
import numpy as np
import lmfit
from collections import Counter, defaultdict
from lmfit.model import ModelResult

from .oc import OC, Parameter, Linear, Quadratic, Keplerian, Sinusoidal
from .model_oc import ModelComponentModel


def _ensure_param(x, *, default: Parameter) -> Parameter:
    """
    Ensure that an input is a `Parameter` object, wrapping scalars if necessary.

    This utility function is used internally to standardize parameter inputs
    for model fitting. If the input is already a `Parameter`, it is returned
    unchanged. If the input is `None`, the provided default is returned. If
    the input is a numeric scalar, it is wrapped into a new `Parameter`.

    Parameters
    ----------
    x : Parameter, float, or None
        Input value to ensure as a `Parameter` object.
    default : Parameter
        Default `Parameter` to use if `x` is None.

    Returns
    -------
    Parameter
        A `Parameter` object representing the input value or default.

    Examples
    --------
    >>> p = Parameter(value=0.5)
    >>> _ensure_param(p, default=Parameter(value=0.0))
    Parameter(value=0.5)
    >>> _ensure_param(1.2, default=Parameter(value=0.0))
    Parameter(value=1.2)
    >>> _ensure_param(None, default=Parameter(value=0.0))
    Parameter(value=0.0)

    Notes
    -----
    - This function is mainly used within `OCLMFit` to unify parameter handling
      for different model components.
    """
    if isinstance(x, Parameter):
        return x
    if x is None:
        return default
    return Parameter(value=x)


class OCLMFit(OC):
    """
    O–C data handler with fitting capabilities using `lmfit`.

    This class extends `OC` by providing methods to fit O–C data to linear,
    quadratic, sinusoidal, Keplerian, and combined models using `lmfit.Model`.
    It supports parameter constraints, fixed parameters, and custom initial values.
    Fitted models can be used to compute residuals or generate new O–C predictions.

    Attributes
    ----------
    math : module
        Mathematical module used in component evaluations. Defaults to `numpy`.

    Notes
    -----
    - Each fit method (`fit_linear`, `fit_quadratic`, `fit_sinusoidal`, etc.) returns
      an `lmfit.ModelResult` with the best-fit parameters.
    - Parameters can be passed as `Parameter` instances, floats, or None. None defaults are used.
    - Residuals can be computed using `residue()` to generate a new `OCLMFit` object.
    - Designed for O–C analysis of periodic phenomena in astronomy, such as eclipsing binaries.

    Examples
    --------
    Fit a linear model to the O–C data:

    >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
    >>> result = oc_fit.fit_linear(a=0.0, b=0.0)

    Fit a Keplerian model with custom initial parameters:

    >>> result = oc_fit.fit_keplerian(amp=0.01, e=0.1, P=1000.0, T0=2450000.0)

    Compute residuals after fitting:

    >>> oc_resid = oc_fit.residue(result)
    """
    math = np

    def fit(
            self,
            model_components: list[ModelComponentModel],
            *,
            nan_policy: str = "raise",
            method: str = "leastsq",
            **kwargs
    ) -> ModelResult:
        """
        Fit the O–C data using one or more model components via `lmfit`.

        This method combines one or more `ModelComponentModel` instances into a single
        composite `lmfit` model, applies parameter constraints, and fits the O–C data
        (`cycle` vs `oc`) according to the provided weights.

        Parameters
        ----------
        model_components : list of ModelComponentModel
            A list of model components (e.g., `Linear`, `Quadratic`, `Keplerian`, `Sinusoidal`)
            to fit simultaneously.
        nan_policy : {'raise', 'omit', 'propagate'}, optional
            How to handle NaN values in the data. Default is 'raise'.
        method : str, optional
            The fitting algorithm to use. Passed directly to `lmfit.Model.fit`. Default is 'leastsq'.
        **kwargs
            Additional keyword arguments forwarded to `lmfit.Model.fit`.

        Returns
        -------
        ModelResult
            The `lmfit.ModelResult` object containing best-fit parameters,
            uncertainties, and fit statistics.

        Raises
        ------
        ValueError
            If the `weights` column contains NaN values.
        TypeError
            If any model component is not compatible with `lmfit`.

        Notes
        -----
        - Each component can have initial parameter values, bounds (`min`, `max`), and fixed status.
        - The method automatically generates unique prefixes for each component to avoid parameter name collisions.
        - Fitted parameters are applied to the respective `ModelComponentModel` instances internally.
        - Supports multiple components of the same type with unique prefixes.

        Examples
        --------
        Fit a linear and sinusoidal model simultaneously:

        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> linear_comp = Linear(a=0.0, b=0.0)
        >>> sinusoidal_comp = Sinusoidal(amp=0.01, P=1000)
        >>> result = oc_fit.fit([linear_comp, sinusoidal_comp], method='leastsq')

        Access fitted parameters:

        >>> result.params['linear_a'].value
        >>> result.params['sinusoidal_amp'].value
        """

        x = np.asarray(self.data["cycle"].to_numpy(), dtype=float)
        y = np.asarray(self.data["oc"].to_numpy(), dtype=float)

        comps = model_components

        for c in comps:
            if hasattr(c, "set_math"):
                c.set_math(self.math)

        def base_name(c):
            return getattr(c, "name", c.__class__.__name__.lower())

        totals = Counter(base_name(c) for c in comps)
        seen = defaultdict(int)

        prefixes = []
        for c in comps:
            b = base_name(c)
            seen[b] += 1
            prefixes.append(f"{b}_" if totals[b] == 1 else f"{b}{seen[b]}_")

        def make_model(comp, prefix) -> lmfit.Model:
            model = lmfit.Model(comp.model_func, independent_vars=["x"], prefix=prefix)
            for p in getattr(comp, "params", {}).keys():
                if p not in model.param_names:
                    model.set_param_hint(p)
            return model

        model = make_model(comps[0], prefixes[0])
        for c, pref in zip(comps[1:], prefixes[1:]):
            model = model + make_model(c, pref)

        params = model.make_params()
        for comp, pref in zip(comps, prefixes):
            cparams = getattr(comp, "params", {}) or {}
            for short_key, cfg in cparams.items():
                full_key = f"{pref}{short_key}"
                if full_key not in params:
                    continue
                p = params[full_key]
                if cfg.value is not None:
                    p.set(value=cfg.value)
                if cfg.min is not None:
                    p.set(min=cfg.min)
                if cfg.max is not None:
                    p.set(max=cfg.max)
                p.set(vary=not bool(cfg.fixed))

        weights = self.data["weights"].to_numpy(dtype=float)
        if np.isnan(weights).any():
            raise ValueError("OCLMFit.fit(...) found NaN values in 'weights'. Please fill or drop them.")

        return model.fit(
            y, params, x=x,
            nan_policy=nan_policy,
            method=method,
            weights=weights,
            **kwargs,
        )

    def residue(self, coefficients: ModelResult, *, x_col: str = "cycle", y_col: str = "oc") -> Self:
        """
        Compute the residual O–C values after subtracting a fitted model.

        This method evaluates the fitted model at the observed cycles and subtracts
        the model values from the observed O–C data, returning a new `OCLMFit`
        instance containing only the residuals.

        Parameters
        ----------
        coefficients : ModelResult
            The fitted `lmfit.ModelResult` object containing the best-fit parameters
            for the model.
        x_col : str, optional
            Name of the column in `self.data` to use as the independent variable.
            Default is 'cycle'.
        y_col : str, optional
            Name of the column in `self.data` to use as the dependent variable (O–C values).
            Default is 'oc'.

        Returns
        -------
        OCLMFit
            A new `OCLMFit` instance with the same metadata as the original object,
            but with the `oc` column replaced by residuals (observed minus fitted values).

        Raises
        ------
        ValueError
            If `x_col` or `y_col` is not present in the data.

        Notes
        -----
        - The returned object preserves all other columns (`minimum_time`, `weights`, `labels`, etc.)
          from the original dataset.
        - Useful for iterative fitting, diagnostics, or residual analysis.

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_linear(a=0.0, b=0.0)
        >>> residuals = oc_fit.residue(result)
        >>> residuals.data.head()
        """
        x = np.asarray(self.data[x_col].to_numpy(), dtype=float)
        yfit = coefficients.eval(x=x)
        new = OCLMFit(
            minimum_time=self.data["minimum_time"].to_list() if "minimum_time" in self.data else None,
            minimum_time_error=self.data["minimum_time_error"].to_list() if "minimum_time_error" in self.data else None,
            weights=self.data["weights"].to_list() if "weights" in self.data else None,
            minimum_type=self.data["minimum_type"].to_list() if "minimum_type" in self.data else None,
            labels=self.data["labels"].to_list() if "labels" in self.data else None,
            cycle=self.data["cycle"].to_list() if "cycle" in self.data else None,
            oc=(self.data[y_col].to_numpy() - yfit).tolist() if y_col in self.data else None,
        )
        return new

    def fit_linear(self, *, a: Union[Parameter, float, None] = None, b: Union[Parameter, float, None] = None,
                   **kwargs) -> ModelResult:
        """
        Fit a linear model to the O–C data using least-squares optimization.

        The model has the form:

            y = a * x + b

        where `x` is the cycle number and `y` is the O–C value. Both slope (`a`) and
        intercept (`b`) can be provided as fixed values, free parameters, or left to
        be estimated from the data.

        Parameters
        ----------
        a : Parameter, float, or None, optional
            Initial guess or fixed value for the slope. If None, defaults to 0.0.
        b : Parameter, float, or None, optional
            Initial guess or fixed value for the intercept. If None, defaults to 0.0.
        **kwargs
            Additional keyword arguments are passed to the underlying `fit` method,
            e.g., `method`, `nan_policy`, etc.

        Returns
        -------
        ModelResult
            An `lmfit.ModelResult` object containing the optimized parameters, fit
            statistics, and methods to evaluate the model or extract residuals.

        Raises
        ------
        ValueError
            If the O–C data contains NaN values in the `weights` column, which are
            required for weighted fitting.

        Notes
        -----
        - This method internally wraps the `a` and `b` values as `Parameter` objects
          if they are not already, allowing fixed or free fitting.
        - Useful for detecting long-term trends in O–C diagrams.

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_linear(a=0.0, b=0.0)
        >>> print(result.best_values)
        {'a': 0.0012, 'b': -0.03}
        >>> residuals = oc_fit.residue(result)
        """
        a = _ensure_param(a, default=Parameter(value=0.0))
        b = _ensure_param(b, default=Parameter(value=0.0))
        comp = Linear(a=a, b=b)
        return self.fit([comp], **kwargs)

    def fit_quadratic(self, *, q: Union[Parameter, float, None] = None, **kwargs) -> ModelResult:
        """
        Fit a quadratic model to the O–C data using least-squares optimization.

        The model has the form:

            y = q * x^2

        where `x` is the cycle number and `y` is the O–C value. The quadratic coefficient
        `q` can be provided as a fixed value, a free parameter, or left to be estimated
        from the data.

        Parameters
        ----------
        q : Parameter, float, or None, optional
            Initial guess or fixed value for the quadratic coefficient. If None, defaults to 0.0.
        **kwargs
            Additional keyword arguments are passed to the underlying `fit` method,
            e.g., `method`, `nan_policy`, etc.

        Returns
        -------
        ModelResult
            An `lmfit.ModelResult` object containing the optimized parameter `q`, fit
            statistics, and methods to evaluate the model or extract residuals.

        Raises
        ------
        ValueError
            If the O–C data contains NaN values in the `weights` column, which are
            required for weighted fitting.

        Notes
        -----
        - This method internally wraps `q` as a `Parameter` object if it is not already,
          allowing fixed or free fitting.
        - Useful for detecting parabolic trends in O–C diagrams, e.g., due to period changes.

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_quadratic(q=1e-6)
        >>> print(result.best_values)
        {'q': 2.3e-6}
        >>> residuals = oc_fit.residue(result)
        """
        q = _ensure_param(q, default=Parameter(value=0.0))
        comp = Quadratic(q=q)
        return self.fit([comp], **kwargs)

    def fit_parabola(
            self,
            *,
            q: Union[Parameter, float, None] = None,
            a: Union[Parameter, float, None] = None,
            b: Union[Parameter, float, None] = None,
            **kwargs
    ) -> ModelResult:
        """
        Fit a combined quadratic and linear model to the O–C data using least-squares optimization.

        The model has the form:

            y = q * x^2 + a * x + b

        where `x` is the cycle number and `y` is the O–C value. Each parameter (`q`, `a`, `b`)
        can be provided as a fixed value, a free parameter, or left to be estimated from the data.

        Parameters
        ----------
        q : Parameter, float, or None, optional
            Initial guess or fixed value for the quadratic coefficient. Defaults to 0.0 if None.
        a : Parameter, float, or None, optional
            Initial guess or fixed value for the linear coefficient. Defaults to 0.0 if None.
        b : Parameter, float, or None, optional
            Initial guess or fixed value for the constant term. Defaults to 0.0 if None.
        **kwargs
            Additional keyword arguments passed to the underlying `fit` method,
            e.g., `method`, `nan_policy`, `weights`, etc.

        Returns
        -------
        ModelResult
            An `lmfit.ModelResult` object containing optimized parameters `q`, `a`, `b`,
            fit statistics, and methods to evaluate the model or extract residuals.

        Raises
        ------
        ValueError
            If the O–C data contains NaN values in the `weights` column.

        Notes
        -----
        - Internally wraps all input values as `Parameter` objects for consistent handling.
        - Useful for modeling O–C diagrams that exhibit both parabolic trends (period changes)
          and linear trends (systematic offsets).

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_parabola(q=1e-6, a=1e-4, b=0.0)
        >>> print(result.best_values)
        {'q': 2.3e-6, 'a': 1.1e-4, 'b': 0.0}
        >>> residuals = oc_fit.residue(result)
        """
        q = _ensure_param(q, default=Parameter(value=0.0))
        a = _ensure_param(a, default=Parameter(value=0.0))
        b = _ensure_param(b, default=Parameter(value=0.0))
        comp_q = Quadratic(q=q)
        comp_l = Linear(a=a, b=b)
        return self.fit([comp_q, comp_l], **kwargs)

    def fit_lite(
            self,
            *,
            amp: Union[Parameter, float, None] = None,
            e: Union[Parameter, float, None] = None,
            omega: Union[Parameter, float, None] = None,
            P: Union[Parameter, float, None] = None,
            T0: Union[Parameter, float, None] = None,
            **kwargs
    ) -> ModelResult:
        """
        Fit a simplified Keplerian model to O–C data using least-squares optimization.

        This "lite" Keplerian model describes a sinusoidal O–C variation approximating
        the orbital motion of a binary system or planet, with the form:

            y = amp * ( (1-e^2)/(1+e*cos(true_anomaly)) * sin(true_anomaly + omega)/denom + e*sin(omega)/denom )

        where `true_anomaly` is calculated from the mean anomaly using Kepler's equation.

        Parameters
        ----------
        amp : Parameter, float, or None, optional
            Semi-amplitude of the O–C variation. Defaults to 1e-3 if None.
        e : Parameter, float, or None, optional
            Orbital eccentricity (0 ≤ e < 0.95). Defaults to 0.0 if None.
        omega : Parameter, float, or None, optional
            Argument of periastron in degrees. Defaults to 90.0 if None.
        P : Parameter, float, or None, optional
            Orbital period. Defaults to 3000.0 if None.
        T0 : Parameter, float, or None, optional
            Reference time of periastron passage. Defaults to 0.0 if None.
        **kwargs
            Additional keyword arguments passed to the underlying `fit` method,
            such as `method`, `nan_policy`, or `weights`.

        Returns
        -------
        ModelResult
            An `lmfit.ModelResult` object containing optimized parameters
            `amp`, `e`, `omega`, `P`, `T0`, fit statistics, and methods for evaluation
            or residual extraction.

        Raises
        ------
        ValueError
            If the O–C data contains NaN values in the `weights` column.

        Notes
        -----
        - Parameters are internally wrapped as `Parameter` objects if they are not already.
        - Useful for rapid fitting of periodic variations in O–C diagrams without modeling additional
          complexities of full Keplerian motion.

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_lite(amp=0.01, e=0.1, P=5000.0)
        >>> print(result.best_values)
        {'amp': 0.012, 'e': 0.098, 'omega': 90.0, 'P': 4990.2, 'T0': 0.0}
        >>> residuals = oc_fit.residue(result)
        """
        amp = _ensure_param(amp, default=Parameter(value=1e-3, min=0.0))
        e = _ensure_param(e, default=Parameter(value=0.0, min=0.0, max=0.95))
        omega = _ensure_param(omega, default=Parameter(value=90.0))
        P = _ensure_param(P, default=Parameter(value=3000.0, min=1.0))
        T0 = _ensure_param(T0, default=Parameter(value=0.0))

        comp = Keplerian(amp=amp, e=e, omega=omega, P=P, T0=T0)
        return self.fit([comp], **kwargs)

    def fit_keplerian(
            self,
            *,
            amp: Union[Parameter, float, None] = None,
            e: Union[Parameter, float, None] = None,
            omega: Union[Parameter, float, None] = None,
            P: Union[Parameter, float, None] = None,
            T0: Union[Parameter, float, None] = None,
            **kwargs
    ) -> ModelResult:
        """
        Fit a Keplerian model to O–C data by delegating to the `fit_lite` method.

        This method is a convenient alias for fitting standard Keplerian variations
        in O–C diagrams, modeling periodic deviations due to orbital motion.
        Internally, it calls `fit_lite` with the given parameters.

        Parameters
        ----------
        amp : Parameter, float, or None, optional
            Semi-amplitude of the O–C variation. Defaults to 1e-3 if None.
        e : Parameter, float, or None, optional
            Orbital eccentricity (0 ≤ e < 0.95). Defaults to 0.0 if None.
        omega : Parameter, float, or None, optional
            Argument of periastron in degrees. Defaults to 90.0 if None.
        P : Parameter, float, or None, optional
            Orbital period. Defaults to 3000.0 if None.
        T0 : Parameter, float, or None, optional
            Reference time of periastron passage. Defaults to 0.0 if None.
        **kwargs
            Additional keyword arguments passed to `fit_lite` and ultimately to `OCLMFit.fit`,
            such as `method`, `nan_policy`, or `weights`.

        Returns
        -------
        ModelResult
            An `lmfit.ModelResult` object containing optimized parameters `amp`, `e`, `omega`, `P`, `T0`,
            fit statistics, and methods for evaluation or residual extraction.

        Raises
        ------
        ValueError
            If the O–C data contains NaN values in the `weights` column (raised in `fit_lite`).

        Notes
        -----
        - This method provides semantic clarity for users who want a Keplerian fit, but
          it is functionally identical to `fit_lite`.
        - Parameters are internally wrapped as `Parameter` objects if they are not already.

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_keplerian(amp=0.01, e=0.1, P=5000.0)
        >>> print(result.best_values)
        {'amp': 0.012, 'e': 0.098, 'omega': 90.0, 'P': 4990.2, 'T0': 0.0}
        >>> residuals = oc_fit.residue(result)
        """
        return self.fit_lite(amp=amp, e=e, omega=omega, P=P, T0=T0, **kwargs)

    def fit_sinusoidal(
            self,
            *,
            amp: Union[Parameter, float, None] = None,
            P: Union[Parameter, float, None] = None,
            **kwargs
    ) -> ModelResult:
        """
        Fit a sinusoidal model to O–C data.

        This method fits a simple sinusoidal variation to the observed O–C values,
        which can model periodic effects such as light-time effects in binary systems.

        Parameters
        ----------
        amp : Parameter, float, or None, optional
            Amplitude of the sinusoidal O–C variation. Defaults to 1e-3 if None.
        P : Parameter, float, or None, optional
            Period of the sinusoidal variation. Defaults to 3000.0 if None.
        **kwargs
            Additional keyword arguments passed to `OCLMFit.fit`, such as `method`,
            `nan_policy`, or `weights`.

        Returns
        -------
        ModelResult
            An `lmfit.ModelResult` object containing optimized parameters `amp` and `P`,
            fit statistics, and methods for evaluation or residual extraction.

        Raises
        ------
        ValueError
            If the O–C data contains NaN values in the `weights` column.

        Notes
        -----
        - This method internally wraps numerical values into `Parameter` objects if they are not already.
        - Useful for detecting or modeling sinusoidal periodicities in O–C diagrams.

        Examples
        --------
        >>> oc_fit = OCLMFit(oc, minimum_time=times, cycle=cycles, weights=weights)
        >>> result = oc_fit.fit_sinusoidal(amp=0.01, P=5000.0)
        >>> print(result.best_values)
        {'amp': 0.011, 'P': 4985.0}
        >>> residuals = oc_fit.residue(result)
        """
        amp = _ensure_param(amp, default=Parameter(value=1e-3, min=0))
        P = _ensure_param(P, default=Parameter(value=3000.0, min=0))

        comp = Sinusoidal(
            amp=amp,
            P=P,
        )
        return self.fit([comp], **kwargs)
