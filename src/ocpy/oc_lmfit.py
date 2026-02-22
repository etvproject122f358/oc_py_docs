import numpy as np
from lmfit.model import ModelResult

from .oc import OC, Parameter, Linear, Quadratic, Keplerian, Sinusoidal
from .model_oc import ModelComponentModel


def _ensure_param(x, *, default: Parameter) -> Parameter:
    """
    Ensure input is a Parameter instance.

    If ``x`` is ``None``, the provided default is returned.
    """

    if isinstance(x, Parameter):
        return x
    if x is None:
        return default
    return Parameter(value=x)


class OCLMFit(OC):
    """
    O−C analysis using lmfit-based parametric modeling.

    This class extends :class:`OC` by providing least-squares fitting
    of analytical O−C model components using the ``lmfit`` library.
    Multiple model components (e.g. linear, quadratic, sinusoidal,
    Keplerian / LITE) can be combined additively and fitted
    simultaneously.

    The fitting backend supports parameter bounds, fixed parameters,
    and weighting of observations.

    Notes
    -----
    The fitted model has the general form:

    .. math::

        (O - C)(E) = \\sum_k f_k(E; \\theta_k)

    where each :math:`f_k` is a model component with its own
    parameter vector :math:`\\theta_k`.

    This class is intended for deterministic O−C model fitting.
    For Bayesian inference, a probabilistic backend should be used.
    """
    math = np

    def fit(
            self,
            model_components: list[ModelComponentModel],
            *,
            nan_policy: str = "raise",
            method: str = "leastsq",
            **kwargs,
    ) -> ModelResult:
        """
        Fit an additive O−C model composed of multiple components.

        Parameters
        ----------
        model_components
            List of model component instances (e.g. Linear, Quadratic,
            Keplerian, Sinusoidal).
        nan_policy
            Policy for handling NaN values during fitting
            (passed to ``lmfit.Model.fit``).
        method
            Optimization method used by lmfit.
        **kwargs
            Additional keyword arguments forwarded to ``lmfit.Model.fit``.

        Returns
        -------
        lmfit.model.ModelResult
            Fitted model result containing best-fit parameters,
            uncertainties, and diagnostics.

        Notes
        -----
        Parameter names are automatically prefixed to avoid collisions
        when multiple components of the same type are present.
        Observation weights are taken from the ``weights`` column of
        the O−C dataset.
        """
        self.logger.info("Fitting O−C model")

        import lmfit
        import numpy as np
        from collections import Counter, defaultdict

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
            return lmfit.Model(comp.model_func, independent_vars=["x"], prefix=prefix)

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
            self.logger.error("OCLMFit.fit(...) found NaN values in 'weights'. Please fill or drop them.")
            raise ValueError("OCLMFit.fit(...) found NaN values in 'weights'. Please fill or drop them.")

        return model.fit(
            y, params, x=x,
            nan_policy=nan_policy,
            method=method,
            weights=weights,
            **kwargs,
        )

    def residue(self, coefficients: ModelResult, *, x_col: str = "cycle", y_col: str = "oc") -> "OCLMFit":
        """
        Compute residual O−C values after model subtraction.

        Parameters
        ----------
        coefficients
            Fitted model result returned by ``fit``.
        x_col
            Column name used as the independent variable.
        y_col
            Column name of the observed O−C values.

        Returns
        -------
        OCLMFit
            New OCLMFit instance containing residual O−C values.
        """
        self.logger.debug("Computing O−C residuals from fitted model")

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

    def fit_linear(self, *, a: Parameter | float | None = None,
                   b: Parameter | float | None = None, **kwargs) -> ModelResult:
        """
        Fit a linear O−C model.

        The model has the form:

        .. math::

            (O - C)(E) = a E + b
        """
        self.logger.info("Fitting O−C model")

        a = _ensure_param(a, default=Parameter(value=0.0))
        b = _ensure_param(b, default=Parameter(value=0.0))
        comp = Linear(a=a, b=b)
        return self.fit([comp], **kwargs)

    def fit_quadratic(self, *, q: Parameter | float | None = None, **kwargs) -> ModelResult:
        """
        Fit a quadratic O−C model.

        .. math::

            (O - C)(E) = q E^2
        """
        self.logger.info("Fitting O−C model")

        q = _ensure_param(q, default=Parameter(value=0.0))
        comp = Quadratic(q=q)
        return self.fit([comp], **kwargs)

    def fit_parabola(
            self,
            *,
            q: Parameter | float | None = None,
            a: Parameter | float | None = None,
            b: Parameter | float | None = None,
            **kwargs,
    ) -> ModelResult:
        """
        Fit a combined quadratic + linear O−C model.

        .. math::

            (O - C)(E) = q E^2 + a E + b
        """
        self.logger.info("Fitting O−C model")

        q = _ensure_param(q, default=Parameter(value=0.0))
        a = _ensure_param(a, default=Parameter(value=0.0))
        b = _ensure_param(b, default=Parameter(value=0.0))
        comp_q = Quadratic(q=q)
        comp_l = Linear(a=a, b=b)
        return self.fit([comp_q, comp_l], **kwargs)

    def fit_lite(
            self,
            *,
            amp: Parameter | float | None = None,
            e: Parameter | float | None = None,
            omega: Parameter | float | None = None,
            P: Parameter | float | None = None,
            T0: Parameter | float | None = None,
            **kwargs,
    ) -> ModelResult:
        """
        Fit a Keplerian light-time effect (LITE) O−C model.

        This model describes timing variations induced by
        orbital motion around a third body.
        """
        self.logger.info("Fitting O−C model")

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
            amp: Parameter | float | None = None,
            e: Parameter | float | None = None,
            omega: Parameter | float | None = None,
            P: Parameter | float | None = None,
            T0: Parameter | float | None = None,
            **kwargs,
    ) -> ModelResult:
        """
        Fit a Keplerian light-time effect (LITE) O−C model.

        This model describes timing variations induced by
        orbital motion around a third body.
        """
        self.logger.info("Fitting O−C model")

        return self.fit_lite(amp=amp, e=e, omega=omega, P=P, T0=T0, **kwargs)

    def fit_sinusoidal(
            self,
            *,
            amp: Parameter | float | None = None,
            P: Parameter | float | None = None,
            **kwargs,
    ) -> ModelResult:
        """
        Fit a sinusoidal O−C model.

        Typically used as a phenomenological approximation to
        cyclic period variations.

        .. math::

            (O - C)(E) = A \\sin\\left( \\frac{2\\pi E}{P} \\right)
        """
        self.logger.info("Fitting O−C model")

        amp = _ensure_param(amp, default=Parameter(value=1e-3, min=0))
        P = _ensure_param(P, default=Parameter(value=3000.0, min=0))

        comp = Sinusoidal(
            amp=amp,
            P=P,
        )
        return self.fit([comp], **kwargs)
