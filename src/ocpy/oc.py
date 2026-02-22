from logging import Logger
from typing import Dict, Self, Callable, List, Literal

from arviz import InferenceData
from numpy.typing import ArrayLike
from lmfit.model import ModelResult
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
import pymc as pm
import pytensor.tensor as pt

from .custom_types import ArrayReducer, NumberOrParam
from .utils import Fixer
from .model_oc import OCModel, ModelComponentModel, ParameterModel


@dataclass
class Parameter(ParameterModel):
    """
    Numerical model parameter with optional bounds and uncertainty.

    This class represents a scalar parameter used in modeling,
    fitting, or inference procedures (e.g. period, epoch, amplitudes).
    A parameter may be free or fixed, bounded or unbounded, and may
    carry an associated uncertainty or probability distribution.

    Attributes
    ----------
    value
        Current value of the parameter. If ``None``, the value is
        considered undefined and may be initialized by a fitter
        or sampler.
    min
        Lower bound of the parameter. If ``None``, the parameter is
        unbounded from below.
    max
        Upper bound of the parameter. If ``None``, the parameter is
        unbounded from above.
    std
        Standard deviation associated with the parameter value.
        Typically represents a 1σ uncertainty from fitting or a
        prior width in probabilistic models.
    fixed
        If ``True``, the parameter is held constant and excluded
        from optimization or sampling procedures.
    distribution
        Name of the statistical distribution associated with this
        parameter. Common choices include ``"normal"``,
        ``"truncatednormal"``, or ``"uniform"``.

    Notes
    -----
    When bounds (``min``, ``max``) are provided together with a
    probabilistic ``distribution``, the parameter is assumed to
    follow a truncated distribution within the specified limits.

    Fixed parameters are treated as constants and do not contribute
    to the dimensionality of the parameter space in fitting or
    sampling algorithms.
    """
    value: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    fixed: bool | None = False
    distribution: str = "truncatednormal"


class ModelComponent(ModelComponentModel):
    """
    Base class for a parametric model component.

    A model component represents a mathematical contribution to a
    composite model (e.g. linear ephemeris, sinusoidal perturbation,
    light-time effect). Each component owns a set of named parameters
    and provides a callable model function that evaluates its
    contribution.

    The component is designed to be backend-agnostic: numerical
    operations can be delegated to different math modules (e.g.
    NumPy, PyMC, PyTensor) to support both deterministic evaluation
    and probabilistic inference.

    Attributes
    ----------
    params
        Dictionary mapping parameter names to :class:`Parameter`
        instances used by this component.
    logger
        Logger instance used for reporting internal state changes,
        warnings, or diagnostic messages during model evaluation
        or configuration.
    math_class
        Numerical backend used for mathematical operations.
        Defaults to :mod:`numpy`.
    """
    logger: Logger
    params: Dict[str, Parameter]

    math_class = np
    _atan2 = staticmethod(np.arctan2)

    def set_math(self, mathmod):
        """
        Set the numerical backend for mathematical operations.

        This method allows the model component to switch between
        different math modules (e.g. NumPy for numerical evaluation,
        PyMC/PyTensor for symbolic or probabilistic computation).

        Parameters
        ----------
        mathmod
            Module providing mathematical functions such as ``sin``,
            ``cos``, and ``arctan2``.

        Returns
        -------
        ModelComponent
            The component instance, allowing method chaining.

        Notes
        -----
        The ``arctan2`` function is handled explicitly to ensure
        compatibility with symbolic backends where it may be defined
        in different namespaces.
        """
        self.logger.info("Setting math module")

        self.math_class = mathmod
        if pm is not None and mathmod is getattr(pm, "math", None):
            self._atan2 = getattr(pm.math, "arctan2", getattr(pt, "arctan2", np.arctan2))
        else:
            self._atan2 = getattr(mathmod, "arctan2", getattr(mathmod, "atan2", np.arctan2))
        return self

    def model_function(self) -> Callable:
        """
        Return the callable model function for this component.

        The returned function evaluates the mathematical contribution
        of this component using its current parameters and numerical
        backend.

        Returns
        -------
        callable
            The model evaluation function associated with this component.
        """
        self.logger.info("Getting model function")

        return self.model_func

    @staticmethod
    def _param(v: NumberOrParam) -> Parameter:
        """
        Normalize a numeric value or parameter to a :class:`Parameter`.

        This helper ensures that model components can accept either
        raw numeric values or fully defined :class:`Parameter` objects
        when constructing or updating parameters.

        Parameters
        ----------
        v
            A numeric value, ``None``, or an existing :class:`Parameter`.

        Returns
        -------
        Parameter
            A :class:`Parameter` instance representing the input value.

        Notes
        -----
        If a numeric value is provided, it is converted into a
        :class:`Parameter` with the value set and no bounds or
        uncertainty defined.
        """
        if isinstance(v, Parameter):
            return v
        return Parameter(value=None if v is None else float(v))


class Linear(ModelComponent):
    """
    Linear model component.

    This component represents a first-order (linear) contribution
    of the form

    .. math::

        f(x) = a x + b

    where ``x`` is typically the cycle number or epoch index.

    In O−C (Observed minus Calculated) analysis, this term is commonly
    used to model:

    * a correction to the reference period (coefficient ``a``),
    * a correction to the reference epoch (offset ``b``).

    Parameters
    ----------
    a
        Linear coefficient. In timing analysis, this often corresponds
        to a correction in the assumed orbital period.
    b
        Constant offset. In timing analysis, this typically represents
        a correction to the reference epoch.
    name
        Optional name of the component.
    logger
        Optional logger instance.

    Notes
    -----
    This component is backend-agnostic and can be evaluated using
    different numerical backends (NumPy, PyMC, PyTensor) depending
    on the active math context.
    """
    name = "linear"

    def __init__(self, a: NumberOrParam = 1.0, b: NumberOrParam = 0.0, *, name: str | None = None,
                 logger: Logger | None = None) -> None:
        if name is not None:
            self.name = name
        self.params = {"a": self._param(a), "b": self._param(b)}
        self.logger = Fixer.logger(logger, self.__class__.__name__)

    def model_func(self, x, a, b):
        """
        Evaluate the linear model.

        Parameters
        ----------
        x
            Independent variable (e.g. cycle number).
        a
            Linear coefficient.
        b
            Constant offset.

        Returns
        -------
        array-like
            Model values computed as:

            .. math::

                f(x) = a x + b
        """
        self.logger.info("Evaluating Linear model component")

        return a * x + b


class Quadratic(ModelComponent):
    """
    Quadratic model component.

    This component represents a second-order contribution of the form

    .. math::

        f(x) = q x^2

    where ``x`` is typically the cycle number.

    In O−C analysis, a quadratic term is commonly interpreted as
    evidence for a **secular change in the orbital period**.
    The coefficient ``q`` is directly related to the period derivative
    :math:`\\dot{P}`.

    Parameters
    ----------
    q
        Quadratic coefficient. In timing analysis, this parameter
        encodes long-term period evolution.
    name
        Optional name of the component.
    logger
        Optional logger instance.

    Notes
    -----
    For eclipse timing variations, the quadratic coefficient is related
    to the period derivative by:

    .. math::

        q = \\frac{1}{2} P_0 \\dot{P}

    where :math:`P_0` is the reference orbital period.

    A positive ``q`` indicates an increasing orbital period,
    while a negative value corresponds to period decay.
    """
    name = "quadratic"

    def __init__(self, q: NumberOrParam = 0.0, *, name: str | None = None, logger: Logger | None = None) -> None:
        if name is not None:
            self.name = name
        self.params = {"q": self._param(q)}
        self.logger = Fixer.logger(logger, self.__class__.__name__)

    def model_func(self, x, q):
        """
        Evaluate the quadratic model.

        Parameters
        ----------
        x
            Independent variable (e.g. cycle number).
        q
            Quadratic coefficient.

        Returns
        -------
        array-like
            Model values computed as:

            .. math::

                f(x) = q x^2
        """
        self.logger.info("Evaluating Quadratic model component")
        return q * (x ** 2)


class Sinusoidal(ModelComponent):
    """
    Sinusoidal model component.

    This component represents a periodic modulation of the form

    .. math::

        f(x) = A \\sin\\left( \\frac{2\\pi x}{P} \\right)

    where ``x`` is typically the cycle number.

    In eclipse timing variation (ETV) and O−C analyses, a sinusoidal
    term is commonly used to model **periodic timing variations**
    caused by effects such as:

    - Light-time effect (LITE) due to a third body
    - Apsidal motion (in simplified form)
    - Magnetic activity cycles (Applegate mechanism)

    Parameters
    ----------
    amp
        Amplitude of the sinusoidal modulation.
        Physically, this corresponds to the maximum timing deviation.
    P
        Period of the sinusoidal modulation (in cycles).
    name
        Optional name of the component.
    logger
        Optional logger instance.

    Notes
    -----
    When interpreted as a light-time effect (LITE), the amplitude
    ``amp`` is related to the projected semi-major axis of the
    eclipsing binary's barycentric motion:

    .. math::

        A = \\frac{a_{12} \\sin i}{c}

    where :math:`a_{12}` is the semi-major axis, :math:`i` the
    inclination of the third-body orbit, and :math:`c` the speed of
    light.

    This component is backend-agnostic and supports NumPy, PyMC,
    and PyTensor math backends.
    """
    name = "sinusoidal"

    def __init__(
            self,
            *,
            amp: NumberOrParam = None,
            P: NumberOrParam = None,
            name: str | None = None,
            logger: Logger | None = None
    ) -> None:
        if name is not None:
            self.name = name

        self.params = {
            "amp": self._param(amp),
            "P": self._param(P),
        }
        self.logger = Fixer.logger(logger, self.__class__.__name__)

    def model_func(self, x, amp, P):
        """
        Evaluate the sinusoidal model.

        Parameters
        ----------
        x
            Independent variable (e.g. cycle number).
        amp
            Amplitude of the sinusoidal modulation.
        P
            Period of the modulation.

        Returns
        -------
        array-like
            Model values computed as:

            .. math::

                f(x) = A \\sin\\left( \\frac{2\\pi x}{P} \\right)
        """
        self.logger.info("Evaluating Sinusoidal model component")

        m = self.math_class
        return amp * m.sin(2.0 * np.pi * x / P)


class Keplerian(ModelComponent):
    """
    Keplerian (light-time effect) model component.

    This component models periodic timing variations caused by
    Keplerian motion, most commonly interpreted as the **light-time
    effect (LITE)** induced by a third body orbiting an eclipsing
    binary system.

    The model evaluates the classical Irwin (1952) formulation of
    the light-time effect:

    .. math::

        (O - C)(t) =
        \\frac{a_{12} \\sin i}{\\sqrt{1 - e^2 \\cos^2 \\omega}}
        \\left[
            \\frac{1 - e^2}{1 + e \\cos \\nu}
            \\sin(\\nu + \\omega)
            + e \\sin \\omega
        \\right]

    where the true anomaly :math:`\\nu` is obtained by solving
    Kepler's equation.

    Parameters
    ----------
    amp
        Amplitude of the light-time effect. Physically corresponds
        to :math:`a_{12} \\sin i / c`.
    e
        Orbital eccentricity of the third-body orbit.
    omega
        Argument of periastron (in degrees).
    P
        Orbital period of the third body.
    T0
        Time of periastron passage.
    name
        Optional name of the component.
    logger
        Optional logger instance.

    Notes
    -----
    The mean anomaly is defined as:

    .. math::

        M = 2\\pi \\frac{t - T_0}{P}

    Kepler's equation is solved iteratively:

    .. math::

        M = E - e \\sin E

    and the true anomaly is computed from the eccentric anomaly:

    .. math::

        \\nu = 2 \\arctan\\left(
        \\sqrt{\\frac{1+e}{1-e}} \\tan\\frac{E}{2}
        \\right)

    This formulation supports both circular and eccentric orbits
    and is backend-agnostic (NumPy, PyMC, PyTensor).
    """
    name = "keplerian"

    def __init__(
            self,
            *,
            amp: NumberOrParam = None,
            e: NumberOrParam = 0.0,
            omega: NumberOrParam = 0.0,
            P: NumberOrParam = None,
            T0: NumberOrParam = None,
            name: str | None = None,
            logger: Logger | None = None
    ) -> None:
        if name is not None:
            self.name = name
        self.params = {
            "amp": self._param(amp),
            "e": self._param(e),
            "omega": self._param(omega),
            "P": self._param(P),
            "T0": self._param(T0),
        }
        self.logger = Fixer.logger(logger, self.__class__.__name__)

    def _wrap_to_pi(self, M):
        """
        Wrap angles to the interval [-pi, pi].

        Parameters
        ----------
        M
            Angle or array of angles in radians.

        Returns
        -------
        array-like
            Angle wrapped to the interval [-pi, pi].

        Notes
        -----
        This implementation uses ``atan2(sin M, cos M)`` to ensure
        numerical stability and backend compatibility.
        """
        self.logger.info("Wrapping angle to [-pi, pi]")

        m = self.math_class
        return self._atan2(m.sin(M), m.cos(M))

    def _kepler_solve(self, M, e, n_iter: int = 5):
        """
        Solve Kepler's equation using Newton–Raphson iteration.

        Parameters
        ----------
        M
            Mean anomaly.
        e
            Orbital eccentricity.
        n_iter
            Number of Newton–Raphson iterations.

        Returns
        -------
        array-like
            Eccentric anomaly.

        Notes
        -----
        Kepler's equation is given by:

        .. math::

            M = E - e \\sin E

        This method converges rapidly for typical eccentricities
        encountered in eclipse timing variation studies.
        """
        self.logger.info("Solving Kepler equation via Newton–Raphson")

        m = self.math_class
        E = M
        for _ in range(n_iter):
            f_val = E - e * m.sin(E) - M
            f_der = 1.0 - e * m.cos(E)
            E = E - f_val / f_der
        return E

    def model_func(self, x, amp, e, omega, P, T0):
        """
        Evaluate the Keplerian light-time effect model.

        Parameters
        ----------
        x
            Independent variable (e.g. cycle number or time).
        amp
            Light-time effect amplitude.
        e
            Orbital eccentricity.
        omega
            Argument of periastron (degrees).
        P
            Orbital period.
        T0
            Time of periastron passage.

        Returns
        -------
        array-like
            Keplerian O−C contribution.
        """
        self.logger.info("Evaluating Keplerian model component")

        m = self.math_class

        w_rad = omega * (np.pi / 180.0)
        M = 2.0 * np.pi * (x - T0) / P
        E = self._kepler_solve(M, e)

        sqrt_term = m.sqrt((1.0 + e) / (1.0 - e))
        tan_half_E = m.tan(E / 2.0)
        true_anom = 2.0 * m.arctan(sqrt_term * tan_half_E)

        denom_factor = m.sqrt(1.0 - (e ** 2) * (m.cos(w_rad)) ** 2)
        amp_term = amp / denom_factor

        term1 = ((1.0 - e ** 2) / (1.0 + e * m.cos(true_anom))) * m.sin(true_anom + w_rad)
        term2 = e * m.sin(w_rad)

        return amp_term * (term1 + term2)


class KeplerianOld(ModelComponent):
    """
    Legacy Keplerian model component for O−C analysis.

    This class implements an older formulation of the Keplerian
    (light-time effect) model used to describe periodic O−C
    variations caused by an orbiting third body.

    The implementation solves Kepler's equation using a fixed
    number of Newton–Raphson iterations and evaluates the timing
    signal directly from the eccentric anomaly.

    Notes
    -----
    - This class is retained for backward compatibility and
      reproducibility of earlier results.
    - New analyses should generally prefer :class:`Keplerian`,
      which provides improved numerical stability and a more
      transparent formulation.
    - The name ``KeplerianOld`` reflects the historical nature
      of this implementation.
    """

    name = "keplerian"

    def __init__(
            self,
            *,
            amp: NumberOrParam = None,
            e: NumberOrParam = 0.0,
            omega: NumberOrParam = 0.0,
            P: NumberOrParam = None,
            T0: NumberOrParam = None,
            name: str | None = None,
            logger: Logger | None = None,
    ) -> None:
        """
        Initialize a legacy Keplerian O−C model component.

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
            Optional component name. Overrides the default name.
        logger
            Optional logger instance. If not provided, a module-level
            logger is created.

        Notes
        -----
        All parameters may be provided either as numeric values or
        as :class:`Parameter` instances for Bayesian inference.
        """
        if name is not None:
            self.name = name
        self.params = {
            "amp": self._param(amp),
            "e": self._param(e),
            "omega": self._param(omega),
            "P": self._param(P),
            "T0": self._param(T0),
        }
        self.logger = Fixer.logger(logger, self.__class__.__name__)

    def _wrap_to_pi(self, M):
        """
        Wrap an angle to the interval (−π, π].

        Parameters
        ----------
        M
            Angle or array of angles (radians).

        Returns
        -------
        array-like
            Angle wrapped to the principal interval (−π, π].

        Notes
        -----
        This method uses ``atan2(sin(M), cos(M))`` to ensure numerical
        stability and backend compatibility.
        """
        self.logger.debug("Wrapping mean anomaly to (−π, π]")

        m = self.math_class
        return self._atan2(m.sin(M), m.cos(M))

    def _kepler_solve(self, M, e, n_iter: int = 8):
        """
        Solve Kepler's equation for the eccentric anomaly.

        Kepler's equation is given by:

        .. math::

            E - e \\sin E = M

        where ``M`` is the mean anomaly and ``e`` is the eccentricity.

        Parameters
        ----------
        M
            Mean anomaly (radians).
        e
            Orbital eccentricity.
        n_iter
            Number of Newton–Raphson iterations.

        Returns
        -------
        array-like
            Eccentric anomaly ``E``.

        Notes
        -----
        - The mean anomaly is wrapped to (−π, π] before iteration.
        - Eccentricity is clipped to the open interval (0, 1).
        - The initial guess is ``E = M + e sin M``.
        """
        self.logger.debug("Solving Kepler equation")

        m = self.math_class
        M = self._wrap_to_pi(M)
        e = m.clip(e, 0.0, 1.0 - 1e-12)
        E = M + e * m.sin(M)
        for _ in range(n_iter):
            f = E - e * m.sin(E) - M
            fp = 1.0 - e * m.cos(E)
            E = E - f / fp
        return E

    def model_func(self, x, amp, e, omega, P, T0):
        """
        Evaluate the legacy Keplerian O−C model.

        Parameters
        ----------
        x
            Independent variable (cycle number or time).
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

        Returns
        -------
        array-like
            Keplerian O−C contribution evaluated at ``x``.

        Notes
        -----
        - This formulation evaluates the signal directly from the
          eccentric anomaly.
        - The expression is algebraically equivalent to the classical
          light-time effect but differs from newer implementations
          in numerical structure.
        """
        self.logger.info("Evaluating legacy Keplerian O−C model")

        m = self.math_class
        wr = omega * (np.pi / 180.0)
        M = 2.0 * np.pi * (x - T0) / P
        E = self._kepler_solve(M, e)

        cosE = m.cos(E)
        sinE = m.sin(E)
        sqrt1me2 = m.sqrt(m.maximum(0.0, 1.0 - e * e))

        return amp * (
                (cosE - e) * m.sin(wr) +
                sqrt1me2 * sinE * m.cos(wr)
        )


class OC(OCModel):
    """
    Observed minus Calculated (O−C) data container.

    This class represents eclipse timing residuals derived from
    observed times of minima and a reference ephemeris. It stores
    O−C values together with auxiliary information such as cycle
    numbers, uncertainties, weights, and minimum classifications.

    The class provides functionality for:
    - binning O−C data
    - recomputing O−C values with different ephemerides
    - fitting analytical timing models (linear, quadratic,
      sinusoidal, Keplerian / LITE)
    - visualization and inference diagnostics

    Notes
    -----
    The O−C residual is defined as:

    .. math::

        (O - C)_i = T_{\\mathrm{obs}, i} - (T_0 + E_i P)

    where:

    - :math:`T_{\\mathrm{obs}, i}` is the observed minimum time
    - :math:`T_0` is the reference epoch
    - :math:`P` is the orbital period
    - :math:`E_i` is the (possibly half-integer) cycle number

    O−C analysis is widely used to study:
    - secular period changes
    - apsidal motion
    - light-time effects (third bodies)
    - dynamical perturbations in eclipsing systems
    """

    def __init__(
            self,
            oc: ArrayLike,
            minimum_time: ArrayLike | None = None,
            minimum_time_error: ArrayLike | None = None,
            weights: ArrayLike | None = None,
            minimum_type: ArrayLike | None = None,
            labels: ArrayLike | None = None,
            cycle: ArrayLike | None = None,
            logger: Logger | None = None,
    ):
        """
        Initialize an O−C dataset.

        Parameters
        ----------
        oc
            Observed minus Calculated residuals.
        minimum_time
            Observed times of minima.
        minimum_time_error
            Uncertainties of minimum times.
        weights
            Statistical weights of observations.
        minimum_type
            Minimum classification (e.g. primary / secondary).
        labels
            Optional labels for data points.
        cycle
            Cycle numbers corresponding to each observation.
        logger
            Optional logger instance.
        """
        ref = minimum_time

        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, ref)
        fixed_weights = Fixer.length_fixer(weights, ref)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, ref)
        fixed_labels_to = Fixer.length_fixer(labels, ref)
        fixed_cycle = Fixer.length_fixer(cycle, ref)
        fixed_oc = Fixer.length_fixer(oc, ref)

        self.data = pd.DataFrame(
            {
                "minimum_time": ref,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
                "cycle": fixed_cycle,
                "oc": fixed_oc,
            }
        )

        self.logger = Fixer.logger(logger, self.__class__.__name__)

    @classmethod
    def from_file(cls, file: str | Path, columns: Dict[str, str] | None = None) -> "OC":
        """
        Load O−C data from a file.

        Supported formats are CSV and Excel.

        Parameters
        ----------
        file : str or Path
            Path to the input file.
        columns : dict, optional
            Mapping for column renaming. Keys are internal names,
            values are file column names.

        Returns
        -------
        OC
            A new OC instance populated from the file.
        """
        file_path = Path(file)
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Use `csv`, `xls`, or `xlsx` instead")

        expected = ["minimum_time", "minimum_time_error", "weights", "minimum_type", "labels", "cycle", "oc"]
        if columns:
            if any(k in expected for k in columns.keys()):
                rename_map = {v: k for k, v in columns.items()}
            else:
                rename_map = columns
            df = df.rename(columns=rename_map)

        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}
        return cls(**kwargs)

    def __str__(self) -> str:
        self.logger.debug("Getting string representation")

        return self.data.__str__()

    def __getitem__(self, item):
        """
        Return a subset of the O−C data.

        If a string is passed, the corresponding column from the underlying
        DataFrame is returned. If an integer or slice is passed, a new
        OC instance containing the subset is returned.

        Parameters
        ----------
        item : str, int, slice, or mask
            The selector for the subset.

        Returns
        -------
        OC or pd.Series
            Subset of the data.
        """
        self.logger.debug("Get item from the data")

        if isinstance(item, str):
            return self.data[item]

        cls = self.__class__

        if isinstance(item, int):
            row = self.data.iloc[item]
            return cls(
                minimum_time=[row.get("minimum_time")],
                minimum_time_error=[row.get("minimum_time_error")],
                weights=[row.get("weights")],
                minimum_type=[row.get("minimum_type")],
                labels=[row.get("labels")],
                cycle=[row.get("cycle")] if "cycle" in self.data.columns else None,
                oc=[row.get("oc")] if "oc" in self.data.columns else None,
            )

        filtered = self.data[item]
        return cls(
            minimum_time=filtered["minimum_time"].tolist(),
            minimum_time_error=filtered[
                "minimum_time_error"].tolist() if "minimum_time_error" in filtered.columns else None,
            weights=filtered["weights"].tolist() if "weights" in filtered.columns else None,
            minimum_type=filtered["minimum_type"].tolist() if "minimum_type" in filtered.columns else None,
            labels=filtered["labels"].tolist() if "labels" in filtered.columns else None,
            cycle=filtered["cycle"].tolist() if "cycle" in filtered.columns else None,
            oc=filtered["oc"].tolist() if "oc" in filtered.columns else None,
        )

    def __setitem__(self, key, value) -> None:
        self.logger.debug("Set item to the data")

        self.data.loc[:, key] = value

    def __len__(self) -> int:
        self.logger.debug("Get length of the data")

        return len(self.data)

    @staticmethod
    def _equal_bins(df: pd.DataFrame, xcol: str, bin_count: int) -> np.ndarray:
        xvals = df[xcol].to_numpy(dtype=float)
        xmin = float(np.min(xvals))
        xmax = float(np.max(xvals))

        bins = np.empty((0, 2), dtype=float)
        total_span = xmax - xmin
        bin_length = total_span / max(1, int(bin_count))

        for k in range(int(bin_count)):
            start = xmin + k * bin_length
            end = xmin + (k + 1) * bin_length if k < bin_count - 1 else xmax
            bins = np.vstack([bins, np.array([[start, end]], dtype=float)])

        return bins

    @staticmethod
    def _smart_bins(
            df: pd.DataFrame,
            xcol: str,
            bin_count: int,
            smart_bin_period: float = 50.0
    ) -> np.ndarray:
        """
        Internal helper for gap-aware binning.

        Identifies large gaps in data and attempts to place bin edges within
        those gaps to avoid splitting clusters of observations.

        Parameters
        ----------
        df : pd.DataFrame
            The data to bin.
        xcol : str
            The name of the x-column.
        bin_count : int
            Target number of bins.
        smart_bin_period : float
            Threshold for identifying a "gap".

        Returns
        -------
        np.ndarray
            Array of bin edges (start, end).
        """
        if smart_bin_period is None or smart_bin_period <= 0:
            raise ValueError("smart_bin_period must be a positive number for _smart_bins")

        df_sorted = df.sort_values(by=xcol)
        xvals = df_sorted[xcol].to_numpy(dtype=float)
        xmin = float(np.min(xvals))
        xmax = float(np.max(xvals))

        bins = np.empty((0, 2), dtype=float)
        bin_start = xmin

        gaps = np.diff(xvals)
        big_gaps = gaps > smart_bin_period
        gap_indexes = np.where(big_gaps)[0]

        for i in gap_indexes:
            bins = np.vstack([bins, np.array([[bin_start, float(xvals[i])]], dtype=float)])
            bin_start = float(xvals[i + 1])

        bins = np.vstack([bins, np.array([[bin_start, xmax]], dtype=float)])

        target_bin_count = int(max(1, bin_count))

        if len(bins) > target_bin_count:
            while len(bins) > target_bin_count:
                inter_gaps = bins[1:, 0] - bins[:-1, 1]
                merge_pos = int(np.argmin(inter_gaps))
                merged_segment = np.array([[bins[merge_pos, 0], bins[merge_pos + 1, 1]]], dtype=float)
                bins = np.vstack([bins[:merge_pos], merged_segment, bins[merge_pos + 2:]])

        if int(bin_count) > len(bins):
            lacking_bins = int(bin_count - len(bins))
            lens = (bins[:, 1] - bins[:, 0]).astype(float)
            weights = lens / np.sum(lens) * lacking_bins
            add_counts = weights.astype(int)
            remainder = lacking_bins - int(np.sum(add_counts))

            if remainder > 0:
                rema = weights % 1.0
                top = np.argsort(-rema)[:remainder]
                add_counts[top] += 1

            new_bins = np.empty((0, 2), dtype=float)
            for i, (start, end) in enumerate(bins):
                k = int(add_counts[i])
                if k <= 0:
                    new_bins = np.vstack([new_bins, np.array([[start, end]], dtype=float)])
                else:
                    edges = np.linspace(start, end, k + 2)
                    segs = np.column_stack([edges[:-1], edges[1:]])
                    new_bins = np.vstack([new_bins, segs])

            bins = new_bins

        return bins

    def bin(
            self,
            bin_count: int = 1,
            bin_method: ArrayReducer | None = None,
            bin_error_method: ArrayReducer | None = None,
            bin_style: Callable[[pd.DataFrame, int], np.ndarray] | None = None,
    ) -> Self:
        """
        Bin the O−C data into a smaller number of points.

        Parameters
        ----------
        bin_count : int
            Target number of bins.
        bin_method : ArrayReducer, optional
            Function to compute the binned value (default is weighted mean).
        bin_error_method : ArrayReducer, optional
            Function to compute the binned error (default is 1/sqrt(sum(weights))).
        bin_style : callable, optional
            Function that generates bin edges. If None, equal-length bins are used.

        Returns
        -------
        OC
            A new OC instance containing the binned data.
        """
        self.logger.debug("Binning O−C data")

        if "cycle" in self.data.columns:
            xcol = "cycle"
        else:
            self.logger.error("`OC.bin` needs or 'cycle' column as x-axis.")
            raise ValueError("`OC.bin` needs or 'cycle' column as x-axis.")

        if "oc" not in self.data.columns:
            self.logger.error("`oc` column is required")
            raise ValueError("`oc` column is required")

        if "weights" not in self.data.columns:
            self.logger.error("`weights` column is required")
            raise ValueError("`weights` column is required")

        if self.data["weights"].hasnans:
            self.logger.error("`weights` contains NaN values")
            raise ValueError("`weights` contain NaN values")

        if self.data[xcol].hasnans:
            self.logger.error(f"`{xcol}` contain NaN values")
            raise ValueError(f"`{xcol}` contain NaN values")

        def mean_binner(array: np.ndarray, weights: np.ndarray) -> float:
            return float(np.average(array, weights=weights))

        def error_binner(weights: np.ndarray) -> float:
            return float(1.0 / np.sqrt(np.sum(weights)))

        if bin_method is None:
            bin_method = mean_binner
        if bin_error_method is None:
            bin_error_method = error_binner

        if bin_style is None:
            bins = self._equal_bins(self.data, xcol, int(bin_count))
        else:
            bins = bin_style(self.data, int(bin_count))

        binned_x: List[float] = []
        binned_ocs: List[float] = []
        binned_errors: List[float] = []

        n_bins = len(bins)
        for i, (start, end) in enumerate(bins):
            if i < n_bins - 1:
                mask = (self.data[xcol] >= start) & (self.data[xcol] < end)
            else:
                mask = (self.data[xcol] >= start) & (self.data[xcol] <= end)

            if not np.any(mask):
                continue

            w = self.data["weights"][mask].to_numpy(dtype=float)
            xarray = self.data[xcol][mask].to_numpy(dtype=float)
            ocarray = self.data["oc"][mask].to_numpy(dtype=float)

            binned_x.append(bin_method(xarray, w))
            binned_ocs.append(bin_method(ocarray, w))
            binned_errors.append(bin_error_method(w))

        new_df = pd.DataFrame()
        new_df["minimum_time"] = np.nan
        new_df["minimum_time_error"] = binned_errors
        new_df["weights"] = np.nan
        new_df["minimum_type"] = None
        new_df["labels"] = "Binned"
        new_df["oc"] = binned_ocs

        new_df["cycle"] = binned_x

        cls = self.__class__
        return cls(
            minimum_time=new_df["minimum_time"].tolist(),
            minimum_time_error=new_df["minimum_time_error"].tolist(),
            weights=new_df["weights"].tolist(),
            minimum_type=new_df["minimum_type"].tolist(),
            labels=new_df["labels"].tolist(),
            cycle=new_df["cycle"].tolist() if "cycle" in new_df.columns else None,
            oc=new_df["oc"].tolist(),
        )

    def merge(self, oc: Self) -> Self:
        """
        Merge this O−C dataset with another.

        Parameters
        ----------
        oc : OC
            The other O−C instance.

        Returns
        -------
        OC
            A new OC instance containing concatenated rows.
        """
        self.logger.info("Merging O−C datasets")

        from copy import deepcopy
        new_oc = deepcopy(self)
        new_oc.data = pd.concat([self.data, oc.data], ignore_index=True, sort=False)
        return new_oc

    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit_model") -> Self:
        """
        Recompute O−C residuals using a linear ephemeris.

        Parameters
        ----------
        reference_minimum
            Reference epoch :math:`T_0`.
        reference_period
            Orbital period :math:`P`.

        Returns
        -------
        OC
            New O−C object with updated residuals.

        Notes
        -----
        The cycle number is computed as:

        .. math::

            E_i = \\mathrm{round}\\left( \\frac{T_{\\mathrm{obs}, i} - T_0}{P} \\right)

        Secondary minima are shifted by half a cycle when
        ``minimum_type`` indicates a secondary eclipse.
        """
        self.logger.info("Recomputing O−C residuals with reference ephemeris")

        import numpy as np

        df = self.data.copy()
        t = np.asarray(df["minimum_time"].to_numpy(), dtype=float)
        phase = (t - reference_minimum) / reference_period
        cycle = np.rint(phase)

        if "minimum_type" in df.columns:
            vals = df["minimum_type"].to_numpy()
            sec = np.zeros_like(t, dtype=bool)
            for i, v in enumerate(vals):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                s = str(v).strip().lower()
                if s in {"1", "ii", "sec", "secondary", "s"} or "ii" in s:
                    sec[i] = True
                elif s in {"0", "i", "pri", "primary", "p"}:
                    sec[i] = False
                else:
                    try:
                        n = int(s)
                        sec[i] = (n == 2)
                    except Exception:
                        pass
            if np.any(sec):
                cycle_sec = np.rint(phase - 0.5) + 0.5
                cycle = np.where(sec, cycle_sec, cycle)

        calculated = reference_minimum + cycle * reference_period
        oc = (t - calculated).astype(float).tolist()

        new_data = {
            "minimum_time": df["minimum_time"].tolist(),
            "minimum_time_error": df["minimum_time_error"].tolist() if "minimum_time_error" in df else None,
            "weights": df["weights"].tolist() if "weights" in df else None,
            "minimum_type": df["minimum_type"].tolist() if "minimum_type" in df else None,
            "labels": df["labels"].tolist() if "labels" in df else None,
        }

        if model_type == "lmfit_model":
            try:
                from .oc_lmfit import OCLMFit
                Target = OCLMFit
            except Exception:
                self.logger.info("Cannot get OCLMFit. Using OC instead")
                Target = OC
        else:
            Target = OC

        return Target(
            minimum_time=new_data["minimum_time"],
            minimum_time_error=new_data["minimum_time_error"],
            weights=new_data["weights"],
            minimum_type=new_data["minimum_type"],
            labels=new_data["labels"],
            cycle=cycle,
            oc=oc,
        )

    def residue(self, coefficients: "ModelResult") -> Self:
        """
        Compute residual O−C values after model subtraction.

        Must be implemented by subclasses (e.g., OCLMFit).

        Parameters
        ----------
        coefficients : ModelResult
            Fitted model parameters.

        Returns
        -------
        OC
            New instance containing residuals.
        """
        self.logger.info("calculating residue")
        pass

    def fit(self, functions: List[ModelComponentModel], ModelComponentModel) -> "ModelResult":
        self.logger.info("Fitting O−C model")
        pass

    def fit_keplerian(
            self,
            *,
            amp: ParameterModel | None = None,
            e: ParameterModel | None = None,
            omega: ParameterModel | None = None,
            P: ParameterModel | None = None,
            T: ParameterModel | None = None,
    ) -> ModelComponentModel:
        self.logger.info("Fitting O−C model")
        pass

    def fit_lite(
            self,
            *,
            amp: ParameterModel | None = None,
            e: ParameterModel | None = None,
            omega: ParameterModel | None = None,
            P: ParameterModel | None = None,
            T: ParameterModel | None = None,
    ) -> ModelComponentModel:
        self.logger.info("Fitting O−C model")
        pass

    def fit_linear(
            self,
            *,
            a: ParameterModel | None = None,
            b: ParameterModel | None = None,
    ) -> ModelComponentModel:
        self.logger.info("Fitting O−C model")
        pass

    def fit_quadratic(
            self,
            *,
            q: ParameterModel | None = None,
    ) -> ModelComponentModel:
        self.logger.info("Fitting O−C model")
        pass

    def fit_sinusoidal(
            self,
            *,
            amp: ParameterModel | None = None,
            P: ParameterModel | None = None,
    ) -> ModelComponentModel:
        """
        Fit a sinusoidal O−C model.

        Must be implemented by subclasses.
        """
        self.logger.info("Fitting O−C model")
        pass

    def fit_parabola(
            self,
            *,
            q: ParameterModel | None = None,
            a: ParameterModel | None = None,
            b: ParameterModel | None = None,
    ) -> ModelComponentModel:
        self.logger.info("Fitting O−C model")
        pass

    def plot(
            self,
            model: InferenceData | ModelResult | List[ModelComponent] = None,
            *,
            ax=None,
            ax_res=None,
            residuals: bool = True,
            title: str | None = None,
            x_col: str = "cycle",
            y_col: str = "oc",
            fig_size: tuple[int, int] = (10, 7),
            plot_kwargs: dict | None = None,
            extension_factor: float = 0.05
    ):
        self.logger.info("Generating O−C plot")
        from .visualization import Plot
        return Plot.plot(
            self,
            model=model,
            ax=ax,
            ax_res=ax_res,
            residuals=residuals,
            title=title,
            x_col=x_col,
            y_col=y_col,
            fig_size=fig_size,
            plot_kwargs=plot_kwargs,
            extension_factor=extension_factor
        )

    def corner(self, model: InferenceData, cornerstyle: Literal["corner", "arviz"] = "corner",
               units: Dict[str, str] | None = None, **kwargs):
        self.logger.info("Ploting corner")

        from .visualization import Plot
        return Plot.plot_corner(model, cornerstyle=cornerstyle, units=units, **kwargs)

    def trace(self, model: InferenceData, **kwargs):
        self.logger.info("Ploting trace")

        from .visualization import Plot
        return Plot.plot_trace(model, **kwargs)
