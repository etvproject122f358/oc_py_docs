from typing import Union, Optional, Dict, Self, Callable, List, Tuple

import pandas
from arviz import InferenceData
from numpy.typing import ArrayLike
from lmfit.model import ModelResult
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .custom_types import ArrayReducer, NumberOrParam
from .utils import Fixer
from .model_oc import OCModel, ModelComponentModel, ParameterModel
from dataclasses import dataclass

try:
    import pymc as pm
    import pytensor.tensor as pt

    _HAS_PYMC = True
except ImportError:
    pm = None
    pt = None
    _HAS_PYMC = False


@dataclass
class Parameter(ParameterModel):
    """
    Represents a model parameter with optional bounds, uncertainty, and distribution.

    Parameters
    ----------
    value : float, optional
        The current or initial value of the parameter.
    min : float, optional
        Lower bound for the parameter, if applicable.
    max : float, optional
        Upper bound for the parameter, if applicable.
    std : float, optional
        Standard deviation of the parameter, used for uncertainty.
    fixed : bool, default=False
        Whether the parameter is fixed (not free to vary) during fitting.
    distribution : str, default='truncatednormal'
        The assumed probability distribution for Bayesian sampling
        (e.g., 'truncatednormal', 'uniform', 'normal').

    Notes
    -----
    - This class inherits from :class:`ParameterModel`.
    - Can be used to define parameters for any :class:`ModelComponent`.
    - Supports integration with fitting routines and Bayesian inference.

    Examples
    --------
    Define a free parameter with initial value 1.0:

    >>> p = Parameter(value=1.0)

    Define a bounded parameter with uncertainty:

    >>> p = Parameter(value=0.5, min=0.0, max=1.0, std=0.1)
    """
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    fixed: Optional[bool] = False
    distribution: str = "truncatednormal"


class ModelComponent(ModelComponentModel):
    """
    Base class for a mathematical model component with named parameters.

    This class provides the foundation for any model component used in
    O–C analysis or other curve-fitting tasks. It manages parameters,
    allows updating from inference data, and supports switching between
    numerical backends such as NumPy or PyMC.

    Attributes
    ----------
    params : dict of str -> Parameter
        A dictionary of parameter names to :class:`Parameter` objects.
        These parameters define the model and can be updated during fitting.
    math_class : module
        Numerical backend used for mathematical operations (default: `numpy`).
    _atan2 : callable
        Function used for two-argument arctangent, adjusted to the current
        math backend.

    Methods
    -------
    set_math(mathmod)
        Switch the math backend for calculations (e.g., NumPy or PyMC).
    model_function()
        Return the underlying model function.
    update_parameters(params_dict)
        Update parameter values from a dictionary of {name: value}.
    update_from_idata(inference_data, group='posterior', stat='median')
        Update parameters from a PyMC :class:`InferenceData` object.
    _param(v)
        Convert a number or existing Parameter into a Parameter object.

    Notes
    -----
    - Intended to be subclassed by concrete model components such as
      :class:`Linear`, :class:`Sinusoidal`, or :class:`Keplerian`.
    - Supports both classical and Bayesian fitting workflows.
    - Provides compatibility with different computational backends,
      including NumPy and PyMC (if available).
    - Parameter values can be updated manually or via inference results.

    Examples
    --------
    Create a simple linear model component:

    >>> linear = Linear(a=1.0, b=0.0)
    >>> linear.params
    {'a': Parameter(value=1.0), 'b': Parameter(value=0.0)}

    Switch to a PyMC math backend (if PyMC installed):

    >>> linear.set_math(pm.math)

    Update parameters manually:

    >>> linear.update_parameters({'a': 2.0})
    >>> linear.params['a'].value
    2.0
    """
    params: Dict[str, Parameter]

    math_class = np
    _atan2 = staticmethod(np.arctan2)

    def set_math(self, mathmod) -> Self:
        """
        Set the mathematical backend for model computations.

        This method allows switching the numerical backend used in all
        mathematical operations of the component, including functions like
        `sin`, `cos`, `sqrt`, and `arctan2`. Supports NumPy by default
        and PyMC if available.

        Parameters
        ----------
        mathmod : module
            The module providing mathematical functions (e.g., `numpy`,
            `pymc.math`, or `pytensor.tensor`).

        Returns
        -------
        self : ModelComponent
            Returns the instance itself to allow method chaining.

        Notes
        -----
        - The `_atan2` function is updated to match the chosen backend.
        - If PyMC is available and `mathmod` corresponds to PyMC's math
          module, `_atan2` uses `pm.math.arctan2` or `pytensor.tensor.arctan2`.
        - If the backend does not provide `arctan2`, it falls back to `numpy.arctan2`.
        - This is useful for switching between classical numerical computation
          and probabilistic programming frameworks.

        Examples
        --------
        Use NumPy as the backend (default):

        >>> component.set_math(np)

        Use PyMC backend if installed:

        >>> import pymc as pm
        >>> component.set_math(pm.math)
        """
        self.math_class = mathmod
        if _HAS_PYMC and mathmod is getattr(pm, "math", None):
            self._atan2 = getattr(pm.math, "arctan2", getattr(pt, "arctan2", np.arctan2))
        else:
            self._atan2 = getattr(mathmod, "arctan2", getattr(mathmod, "atan2", np.arctan2))
        return self

    def model_function(self) -> Callable:
        """
        Return the underlying model function of the component.

        The model function defines the mathematical relationship between
        the independent variable(s) and the output, based on the component's
        parameters. It is intended to be used for evaluation, fitting, or
        plotting.

        Returns
        -------
        callable
            The function implementing the model. The callable typically
            has the signature:

            ``f(x, param1, param2, ...)``

            where `x` is the independent variable and the remaining
            arguments are the component's parameters.

        Notes
        -----
        - Subclasses must define the `model_func` method, which this
          property returns.
        - Useful for passing to optimization routines or plotting functions
          that require a callable.

        Examples
        --------
        Retrieve and call the model function of a linear component:

        >>> linear = Linear(a=2.0, b=1.0)
        >>> f = linear.model_function()
        >>> f(3, linear.params['a'].value, linear.params['b'].value)
        7.0
        """
        return self.model_func

    def update_parameters(self, params_dict: Dict[str, float]) -> Self:
        """
        Update the values of existing parameters from a dictionary.

        Parameters
        ----------
        params_dict : dict of str -> float
            Dictionary mapping parameter names to their new numeric values.
            Only parameters that already exist in the component will be updated;
            any unknown keys are ignored.

        Returns
        -------
        self : ModelComponent
            Returns the instance itself to allow method chaining.

        Notes
        -----
        - This method does not create new parameters; it only updates values
          of existing ones.
        - All values are cast to `float` before assignment.
        - Useful for manually adjusting parameters between fitting steps
          or after loading results from inference.

        Examples
        --------
        Update a linear model component's parameters:

        >>> linear = Linear(a=1.0, b=0.0)
        >>> linear.update_parameters({'a': 2.5, 'b': -1.0})
        >>> linear.params['a'].value
        2.5
        >>> linear.params['b'].value
        -1.0

        Attempting to update a non-existent parameter has no effect:

        >>> linear.update_parameters({'c': 10.0})
        >>> 'c' in linear.params
        False
        """
        for k, v in params_dict.items():
            if k in self.params:
                self.params[k].value = float(v)
        return self

    def update_from_idata(self, inference_data, group="posterior", stat="median") -> Self:
        """
        Update model parameters from a PyMC `InferenceData` object.

        This method extracts parameter values from a Bayesian posterior
        (or other group) and updates the component's parameters accordingly.
        It automatically matches variables that are prefixed with the component's
        name.

        Parameters
        ----------
        inference_data : arviz.InferenceData
            The InferenceData object containing posterior samples from a PyMC model.
        group : str, default='posterior'
            The group within the InferenceData to use (e.g., 'posterior', 'prior').
        stat : str, default='median'
            Statistic to use for updating parameters. Options include:
            - 'median': use the median of the posterior samples.
            - 'mean': use the mean of the posterior samples.
            - Any other value: uses the first sample as a fallback.

        Returns
        -------
        self : ModelComponent
            The component instance with updated parameter values.

        Notes
        -----
        - Only parameters present in `self.params` and matching the component's
          prefix are updated.
        - Useful for propagating posterior estimates back into model components
          for further computation or plotting.
        - Requires PyMC and ArviZ to be installed.

        Examples
        --------
        Suppose `idata` contains posterior samples for a linear component
        named 'linear':

        >>> linear = Linear(a=1.0, b=0.0)
        >>> linear.update_from_idata(idata, group='posterior', stat='median')
        >>> linear.params['a'].value  # updated from posterior median
        2.3
        >>> linear.params['b'].value
        -0.1
        """
        prefix = getattr(self, "name", "")

        # Get the variables for this component
        variable_names = [var_name for var_name in inference_data[group].data_vars if var_name.startswith(f"{prefix}_")]

        params_to_update = {}
        for variable_name in variable_names:
            parameter_name = variable_name[len(prefix) + 1:]
            if parameter_name in self.params:
                if stat == "median":
                    value = inference_data[group][variable_name].median(dim=("chain", "draw")).item()
                elif stat == "mean":
                    value = inference_data[group][variable_name].mean(dim=("chain", "draw")).item()
                else:
                    # Fallback to first sample
                    value = inference_data[group][variable_name].values[0, 0]
                params_to_update[parameter_name] = float(value)

        return self.update_parameters(params_to_update)

    @staticmethod
    def _param(v: NumberOrParam) -> Parameter:
        """
        Convert a number or existing Parameter into a Parameter object.

        This utility ensures that all model parameters are represented
        as instances of the :class:`Parameter` class, which is required
        for consistent handling in model components.

        Parameters
        ----------
        v : float or Parameter or None
            The input value to convert. If `v` is already a `Parameter`,
            it is returned unchanged. If `v` is a number, it is wrapped
            in a new `Parameter` with `value=v`. If `v` is `None`, a
            `Parameter` with `value=None` is returned.

        Returns
        -------
        Parameter
            A `Parameter` instance corresponding to the input.

        Notes
        -----
        - This method is used internally when initializing model components
          to normalize parameter inputs.
        - All numeric values are cast to `float` for consistency.

        Examples
        --------
        Convert a float to a Parameter:

        >>> p = ModelComponent._param(2.5)
        >>> isinstance(p, Parameter)
        True
        >>> p.value
        2.5

        Pass an existing Parameter:

        >>> p_existing = Parameter(value=1.0)
        >>> p2 = ModelComponent._param(p_existing)
        >>> p2 is p_existing
        True

        Handle None:

        >>> p_none = ModelComponent._param(None)
        >>> p_none.value is None
        True
        """
        if isinstance(v, Parameter):
            return v
        return Parameter(value=None if v is None else float(v))


class Linear(ModelComponent):
    """
    Linear model component of the form `f(x) = a * x + b`.

    Represents a simple linear relationship between an independent variable
    `x` and the output, parameterized by slope `a` and intercept `b`.

    Parameters
    ----------
    a : float or Parameter, default=1.0
        Slope of the linear model. Can be a numeric value or a `Parameter` object.
    b : float or Parameter, default=0.0
        Intercept of the linear model. Can be a numeric value or a `Parameter` object.
    name : str, optional
        Optional name for the component. Defaults to "linear".

    Attributes
    ----------
    params : dict
        Dictionary of parameters: `{'a': Parameter, 'b': Parameter}`.
    name : str
        Name of the component (default: "linear").

    Methods
    -------
    model_func(x, a, b)
        Evaluate the linear model for input `x` using parameters `a` and `b`.

    Examples
    --------
    Create a linear component with default parameters:

    >>> linear = Linear()
    >>> linear.model_func(3, linear.params['a'].value, linear.params['b'].value)
    3.0

    Create a linear component with custom slope and intercept:

    >>> linear = Linear(a=2.0, b=1.0)
    >>> linear.model_func(3, linear.params['a'].value, linear.params['b'].value)
    7.0
    """
    name = "linear"

    def __init__(self, a: NumberOrParam = 1.0, b: NumberOrParam = 0.0, *, name: Optional[str] = None) -> None:
        """
        Initialize a Linear model component with slope and intercept.

        Parameters
        ----------
        a : float or Parameter, default=1.0
            Slope of the linear function. Can be a numeric value or a `Parameter` object.
        b : float or Parameter, default=0.0
            Intercept of the linear function. Can be a numeric value or a `Parameter` object.
        name : str, optional
            Optional name for the component. If provided, it overrides the default name "linear".

        Notes
        -----
        - All numeric inputs are converted to `Parameter` objects internally.
        - The `params` dictionary is created to store the component parameters.

        Examples
        --------
        Default linear component:

        >>> linear = Linear()
        >>> linear.params['a'].value
        1.0
        >>> linear.params['b'].value
        0.0

        Custom slope and intercept:

        >>> linear = Linear(a=2.0, b=1.0)
        >>> linear.params['a'].value
        2.0
        >>> linear.params['b'].value
        1.0

        Custom name:

        >>> linear = Linear(a=1.0, b=0.0, name='my_linear')
        >>> linear.name
        'my_linear'
        """
        if name is not None:
            self.name = name
        self.params = {"a": self._param(a), "b": self._param(b)}

    def model_func(self, x, a, b) -> float | np.ndarray:
        """
        Evaluate the linear model `f(x) = a * x + b`.

        Parameters
        ----------
        x : float or array-like
            Independent variable(s) at which to evaluate the model.
        a : float
            Slope of the linear function.
        b : float
            Intercept of the linear function.

        Returns
        -------
        float or np.ndarray
            The computed value(s) of the linear function at `x`.

        Examples
        --------
        Evaluate a linear component at a single point:

        >>> linear = Linear(a=2.0, b=1.0)
        >>> linear.model_func(3, linear.params['a'].value, linear.params['b'].value)
        7.0

        Evaluate at multiple points:

        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3])
        >>> linear.model_func(x, linear.params['a'].value, linear.params['b'].value)
        array([1., 3., 5., 7.])
        """
        return a * x + b


class Quadratic(ModelComponent):
    """
    Quadratic model component of the form `f(x) = q * x^2`.

    Represents a simple quadratic relationship between an independent variable
    `x` and the output, parameterized by a single coefficient `q`.

    Parameters
    ----------
    q : float or Parameter, default=0.0
        Quadratic coefficient. Can be a numeric value or a `Parameter` object.
    name : str, optional
        Optional name for the component. Defaults to "quadratic".

    Attributes
    ----------
    params : dict
        Dictionary of parameters: `{'q': Parameter}`.
    name : str
        Name of the component (default: "quadratic").

    Methods
    -------
    model_func(x, q)
        Evaluate the quadratic model for input `x` using parameter `q`.

    Examples
    --------
    Create a quadratic component with default parameter:

    >>> quad = Quadratic()
    >>> quad.model_func(2, quad.params['q'].value)
    0.0

    Create a quadratic component with custom coefficient:

    >>> quad = Quadratic(q=3.0)
    >>> quad.model_func(2, quad.params['q'].value)
    12.0
    """
    name = "quadratic"

    def __init__(self, q: NumberOrParam = 0.0, *, name: Optional[str] = None) -> None:
        """
        Initialize a Quadratic model component with a quadratic coefficient.

        Parameters
        ----------
        q : float or Parameter, default=0.0
            Coefficient of the quadratic term. Can be a numeric value or a `Parameter` object.
        name : str, optional
            Optional name for the component. If provided, it overrides the default name "quadratic".

        Notes
        -----
        - The input `q` is converted to a `Parameter` object internally if it is numeric.
        - The `params` dictionary is created to store the component parameter.

        Examples
        --------
        Default quadratic component:

        >>> quad = Quadratic()
        >>> quad.params['q'].value
        0.0

        Custom coefficient:

        >>> quad = Quadratic(q=2.5)
        >>> quad.params['q'].value
        2.5

        Custom name:

        >>> quad = Quadratic(q=1.0, name='my_quad')
        >>> quad.name
        'my_quad'
        """
        if name is not None:
            self.name = name
        self.params = {"q": self._param(q)}

    def model_func(self, x, q) -> float | np.ndarray:
        """
        Evaluate the quadratic model `f(x) = q * x^2`.

        Parameters
        ----------
        x : float or array-like
            Independent variable(s) at which to evaluate the model.
        q : float
            Coefficient of the quadratic term.

        Returns
        -------
        float or np.ndarray
            The computed value(s) of the quadratic function at `x`.

        Examples
        --------
        Evaluate a quadratic component at a single point:

        >>> quad = Quadratic(q=2.0)
        >>> quad.model_func(3, quad.params['q'].value)
        18.0

        Evaluate at multiple points:

        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3])
        >>> quad.model_func(x, quad.params['q'].value)
        array([ 0.,  2.,  8., 18.])
        """
        return q * (x ** 2)


class Sinusoidal(ModelComponent):
    """
    Sinusoidal model component of the form `f(x) = amp * sin(2π * x / P)`.

    Represents a simple periodic function with amplitude `amp` and period `P`.

    Parameters
    ----------
    amp : float or Parameter, optional
        Amplitude of the sinusoidal function. Can be a numeric value or a `Parameter` object.
    P : float or Parameter, optional
        Period of the sinusoidal function. Can be a numeric value or a `Parameter` object.
    name : str, optional
        Optional name for the component. Defaults to "sinusoidal".

    Attributes
    ----------
    params : dict
        Dictionary of parameters: `{'amp': Parameter, 'P': Parameter}`.
    name : str
        Name of the component (default: "sinusoidal").

    Methods
    -------
    model_func(x, amp, P)
        Evaluate the sinusoidal model for input `x` using parameters `amp` and `P`.

    Examples
    --------
    Create a sinusoidal component with amplitude 1.0 and period 2.0:

    >>> sinus = Sinusoidal(amp=1.0, P=2.0)
    >>> sinus.model_func(0, sinus.params['amp'].value, sinus.params['P'].value)
    0.0

    Evaluate at multiple points:

    >>> import numpy as np
    >>> x = np.array([0, 0.5, 1.0, 1.5])
    >>> sinus.model_func(x, sinus.params['amp'].value, sinus.params['P'].value)
    array([0. , 1. , 0. , -1.])
    """
    name = "sinusoidal"

    def __init__(self, *, amp: NumberOrParam = None, P: NumberOrParam = None, name: Optional[str] = None) -> None:
        """
        Initialize a Sinusoidal model component with amplitude and period.

        Parameters
        ----------
        amp : float or Parameter, optional
            Amplitude of the sinusoidal function. If None, the amplitude is unassigned.
        P : float or Parameter, optional
            Period of the sinusoidal function. If None, the period is unassigned.
        name : str, optional
            Optional name for the component. If provided, it overrides the default name "sinusoidal".

        Notes
        -----
        - Numeric inputs are converted to `Parameter` objects internally.
        - The `params` dictionary stores the component parameters: `amp` and `P`.

        Examples
        --------
        Default sinusoidal component (unassigned parameters):

        >>> sinus = Sinusoidal()
        >>> sinus.params['amp'].value is None
        True
        >>> sinus.params['P'].value is None
        True

        Custom amplitude and period:

        >>> sinus = Sinusoidal(amp=1.0, P=2.0)
        >>> sinus.params['amp'].value
        1.0
        >>> sinus.params['P'].value
        2.0

        Custom name:

        >>> sinus = Sinusoidal(amp=1.0, P=2.0, name='my_sin')
        >>> sinus.name
        'my_sin'
        """
        if name is not None:
            self.name = name

        self.params = {
            "amp": self._param(amp),
            "P": self._param(P),
        }

    def model_func(self, x, amp, P) -> float | np.ndarray:
        """
        Evaluate the sinusoidal model `f(x) = amp * sin(2π * x / P)`.

        Parameters
        ----------
        x : float or array-like
            Independent variable(s) at which to evaluate the model.
        amp : float
            Amplitude of the sinusoidal function.
        P : float
            Period of the sinusoidal function.

        Returns
        -------
        float or np.ndarray
            The computed value(s) of the sinusoidal function at `x`.

        Examples
        --------
        Evaluate at a single point:

        >>> sinus = Sinusoidal(amp=2.0, P=4.0)
        >>> sinus.model_func(1, sinus.params['amp'].value, sinus.params['P'].value)
        2.0

        Evaluate at multiple points:

        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> sinus.model_func(x, sinus.params['amp'].value, sinus.params['P'].value)
        array([ 0.,  2.,  0., -2., -0.])
        """
        m = self.math_class
        return amp * m.sin(2.0 * np.pi * x / P)


class Keplerian(ModelComponent):
    """
    Keplerian orbital model component for radial velocity variations.

    Represents a Keplerian orbit with parameters corresponding to
    the orbital amplitude, eccentricity, argument of periastron,
    orbital period, and time of periastron passage.

    The model computes radial velocity or timing variations as a
    function of time based on classical Keplerian motion.

    Parameters
    ----------
    amp : float or Parameter, optional
        Amplitude of the Keplerian signal.
    e : float or Parameter, default=0.0
        Orbital eccentricity. Must be between 0 and 1.
    omega : float or Parameter, default=0.0
        Argument of periastron in degrees.
    P : float or Parameter, optional
        Orbital period.
    T0 : float or Parameter, optional
        Time of periastron passage.
    name : str, optional
        Optional name for the component. Defaults to "keplerian".

    Attributes
    ----------
    params : dict
        Dictionary of parameters: `{'amp', 'e', 'omega', 'P', 'T0'}`.
    name : str
        Name of the component (default: "keplerian").

    Methods
    -------
    model_func(x, amp, e, omega, P, T0)
        Evaluate the Keplerian model at times `x`.
    _kepler_solve(M, e, n_iter=5)
        Solve Kepler's equation using iterative method.

    Notes
    -----
    - The input angles are in degrees and internally converted to radians.
    - Uses Newton-Raphson iteration to solve Kepler's equation.
    - Suitable for modeling radial velocity or timing variations in binary stars or exoplanets.

    Examples
    --------
    Create a Keplerian component with default parameters:

    >>> kepler = Keplerian()
    >>> kepler.params['e'].value
    0.0

    Custom parameters:

    >>> kepler = Keplerian(amp=2.0, e=0.5, omega=45.0, P=10.0, T0=0.0)
    >>> kepler.params['amp'].value
    2.0
    >>> kepler.params['P'].value
    10.0
    """
    name = "keplerian"

    def __init__(
            self, *,
            amp: NumberOrParam = None,
            e: NumberOrParam = 0.0,
            omega: NumberOrParam = 0.0,
            P: NumberOrParam = None,
            T0: NumberOrParam = None,
            name: Optional[str] = None
    ) -> None:
        """
        Initialize a Keplerian orbital model component.

        Parameters
        ----------
        amp : float or Parameter, optional
            Amplitude of the Keplerian signal.
        e : float or Parameter, default=0.0
            Orbital eccentricity, must be between 0 and 1.
        omega : float or Parameter, default=0.0
            Argument of periastron in degrees.
        P : float or Parameter, optional
            Orbital period.
        T0 : float or Parameter, optional
            Time of periastron passage.
        name : str, optional
            Optional name for the component. If provided, overrides the default "keplerian".

        Notes
        -----
        - All numeric inputs are internally converted to `Parameter` objects.
        - The `params` dictionary stores the component parameters: `amp`, `e`, `omega`, `P`, `T0`.
        - The default iteration count for solving Kepler's equation is set in `_kepler_solve`.

        Examples
        --------
        Default Keplerian component:

        >>> kepler = Keplerian()
        >>> kepler.params['e'].value
        0.0
        >>> kepler.name
        'keplerian'

        Custom parameters:

        >>> kepler = Keplerian(amp=1.5, e=0.3, omega=60.0, P=10.0, T0=0.0)
        >>> kepler.params['amp'].value
        1.5
        >>> kepler.params['P'].value
        10.0

        Custom name:

        >>> kepler = Keplerian(name='orbit1')
        >>> kepler.name
        'orbit1'
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

    def _kepler_solve(self, M, e, n_iter: int = 5) -> float | np.ndarray:
        """
        Solve Kepler's equation for the eccentric anomaly `E` using iterative Newton-Raphson method.

        The equation solved is:

            E - e * sin(E) = M

        where `M` is the mean anomaly and `e` is the eccentricity.

        Parameters
        ----------
        M : float or np.ndarray
            Mean anomaly (radians) at which to solve Kepler's equation.
        e : float
            Orbital eccentricity (0 ≤ e < 1).
        n_iter : int, default=5
            Number of iterations for the Newton-Raphson solver.

        Returns
        -------
        float or np.ndarray
            Eccentric anomaly `E` corresponding to input mean anomaly `M`.

        Notes
        -----
        - This method uses a fixed number of Newton-Raphson iterations.
        - Convergence is generally sufficient for small to moderate eccentricities.
        - For high eccentricities or high precision, increase `n_iter`.

        Examples
        --------
        Solve for a single mean anomaly:

        >>> kepler = Keplerian()
        >>> import numpy as np
        >>> M = np.pi / 2
        >>> e = 0.1
        >>> E = kepler._kepler_solve(M, e)
        >>> isinstance(E, float)
        True

        Solve for an array of mean anomalies:

        >>> M_array = np.array([0.0, np.pi/4, np.pi/2])
        >>> E_array = kepler._kepler_solve(M_array, e)
        >>> E_array.shape
        (3,)
        """
        m = self.math_class
        E = M
        for _ in range(n_iter):
            f_val = E - e * m.sin(E) - M
            f_der = 1.0 - e * m.cos(E)
            E = E - f_val / f_der
        return E

    def model_func(self, x, amp, e, omega, P, T0) -> float | np.ndarray:
        """
        Evaluate the Keplerian model at given times.

        Computes the Keplerian signal (e.g., radial velocity or timing variations)
        based on orbital parameters: amplitude, eccentricity, argument of periastron,
        period, and time of periastron passage.

        Parameters
        ----------
        x : float or array-like
            Times at which to evaluate the model.
        amp : float
            Amplitude of the Keplerian signal.
        e : float
            Orbital eccentricity (0 ≤ e < 1).
        omega : float
            Argument of periastron in degrees.
        P : float
            Orbital period.
        T0 : float
            Time of periastron passage.

        Returns
        -------
        float or np.ndarray
            Keplerian model values at the specified times.

        Notes
        -----
        - The argument of periastron `omega` is internally converted to radians.
        - Uses `_kepler_solve` to determine the eccentric anomaly.
        - Handles both scalar and array inputs for `x`.
        - Suitable for modeling timing variations in eclipsing binaries or radial velocity signals.

        Examples
        --------
        Evaluate at a single time:

        >>> kepler = Keplerian(amp=1.0, e=0.1, omega=30.0, P=10.0, T0=0.0)
        >>> kepler.model_func(5, 1.0, 0.1, 30.0, 10.0, 0.0)
        0.85  # Example value (approximate)

        Evaluate at multiple times:

        >>> import numpy as np
        >>> times = np.linspace(0, 10, 5)
        >>> kepler.model_func(times, 1.0, 0.1, 30.0, 10.0, 0.0)
        array([0.0, 0.52, 0.85, 0.85, 0.52])
        """
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
    name = "keplerian"

    def __init__(
            self,
            *,
            amp: NumberOrParam = None,
            e: NumberOrParam = 0.0,
            omega: NumberOrParam = 0.0,
            P: NumberOrParam = None,
            T0: NumberOrParam = None,
            name: Optional[str] = None,
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

    def _wrap_to_pi(self, M):
        m = self.math_class
        return self._atan2(m.sin(M), m.cos(M))

    def _kepler_solve(self, M, e, n_iter: int = 8):
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
    Observed-minus-Calculated (O–C) data container and analysis class.

    Represents timing variations of observed events (e.g., eclipses, pulsations)
    relative to a reference ephemeris. Provides methods for binning, merging,
    computing O–C values, and fitting models to timing variations.

    Attributes
    ----------
    data : pd.DataFrame
        Internal DataFrame storing columns:
        'minimum_time', 'minimum_time_error', 'weights', 'minimum_type',
        'labels', 'cycle', 'oc'.

    Methods
    -------
    from_file(file, columns=None)
        Create an OC instance from a CSV or Excel file.
    __getitem__(item)
        Access a column, row, or filtered OC subset.
    __setitem__(key, value)
        Assign values to a column.
    __len__()
        Return the number of entries.
    bin(bin_count=1, bin_method=None, bin_error_method=None, bin_style=None)
        Bin O–C data using weighted averages.
    merge(oc)
        Merge with another OC instance.
    calculate_oc(reference_minimum, reference_period, model_type='lmfit')
        Compute O–C values relative to a reference ephemeris.
    plot(model=None, ax=None, res_ax=None, res=True, ...)
        Visualize O–C data and optional model fits.

    Notes
    -----
    - Supports both scalar and array-like input for all columns.
    - `cycle` is automatically computed or can be provided explicitly.
    - Binning can be performed with custom reducers and bin styles.
    - Integration with `ModelComponent` subclasses enables model fitting.

    Examples
    --------
    Create an OC instance from raw data:

    >>> oc = OC(oc=[0.1, -0.05, 0.0],
    ...         minimum_time=[2450000.0, 2450001.0, 2450002.0],
    ...         weights=[1.0, 1.0, 1.0])

    Access O–C column:

    >>> oc['oc']
    0    0.10
    1   -0.05
    2    0.00
    Name: oc, dtype: float64

    Bin O–C data:

    >>> binned = oc.bin(bin_count=2)

    Merge with another OC:

    >>> merged = oc.merge(other_oc)
    """

    def __init__(
            self,
            oc: ArrayLike,
            minimum_time: Optional[ArrayLike] = None,
            minimum_time_error: Optional[ArrayLike] = None,
            weights: Optional[ArrayLike] = None,
            minimum_type: Optional[ArrayLike] = None,
            labels: Optional[ArrayLike] = None,
            cycle: Optional[ArrayLike] = None
    ):
        """
        Initialize an O–C (Observed-minus-Calculated) data object.

        Parameters
        ----------
        oc : array-like
            Observed minus calculated values.
        minimum_time : array-like, optional
            Times of observed events. Required if other columns are provided.
        minimum_time_error : array-like, optional
            Measurement uncertainties for `minimum_time`.
        weights : array-like, optional
            Weights associated with each observation. Typically inverse variance.
        minimum_type : array-like, optional
            Binary type indicators (e.g., primary/secondary eclipse).
        labels : array-like, optional
            Labels for each observation.
        cycle : array-like, optional
            Cycle numbers corresponding to each observation.

        Raises
        ------
        ValueError
            If `minimum_time` is None when required for fixing lengths of other columns.

        Notes
        -----
        - All input arrays are converted to lists and stored in a `pandas.DataFrame`.
        - Lengths of optional arrays are fixed to match `minimum_time` using `Fixer.length_fixer`.
        - Supports both scalar and array-like inputs; scalars are broadcast to match `minimum_time`.

        Examples
        --------
        Create an OC instance with minimal data:

        >>> oc = OC(oc=[0.1, -0.05, 0.0],
        ...         minimum_time=[2450000.0, 2450001.0, 2450002.0])

        Create an OC instance with errors and weights:

        >>> oc = OC(oc=[0.1, -0.05, 0.0],
        ...         minimum_time=[2450000.0, 2450001.0, 2450002.0],
        ...         minimum_time_error=[0.01, 0.02, 0.01],
        ...         weights=[100, 25, 100])
        """
        reference_time = minimum_time

        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, reference_time)
        fixed_weights = Fixer.length_fixer(weights, reference_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, reference_time)
        fixed_labels_to = Fixer.length_fixer(labels, reference_time)
        fixed_cycle = Fixer.length_fixer(cycle, reference_time)
        fixed_oc = Fixer.length_fixer(oc, reference_time)

        self.data = pd.DataFrame(
            {
                "minimum_time": reference_time,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
                "cycle": fixed_cycle,
                "oc": fixed_oc,
            }
        )

    @classmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        """
        Load O–C data from a CSV or Excel file into an `OC` instance.

        Parameters
        ----------
        file : str or Path
            Path to the input file. Supported formats: `.csv`, `.xls`, `.xlsx`.
        columns : dict, optional
            Mapping of file column names to OC attribute names. Keys are original
            column names in the file; values are the corresponding attribute names
            (`minimum_time`, `oc`, `weights`, `cycle`, etc.).

        Returns
        -------
        OC
            An OC instance containing data loaded from the file.

        Raises
        ------
        ValueError
            If the file type is unsupported.

        Notes
        -----
        - Any columns not present in the file are filled with `None`.
        - Column renaming allows flexibility for files with nonstandard column names.
        - The method automatically handles array-like columns and converts them
          for internal storage in a `pandas.DataFrame`.

        Examples
        --------
        Load a CSV file with standard columns:

        >>> oc = OC.from_file("observations.csv")

        Load an Excel file with custom column names:

        >>> oc = OC.from_file("observations.xlsx",
        ...                   columns={"Time": "minimum_time",
        ...                            "O-C": "oc",
        ...                            "Weight": "weights"})
        """
        file_path = Path(file)
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Use `csv`, `xls`, or `xlsx` instead")

        if columns:
            rename_map = {k: v for k, v in columns.items() if k in df.columns}
            df = df.rename(columns=rename_map)

        expected = ["minimum_time", "minimum_time_error", "weights", "minimum_type", "labels", "cycle", "oc"]
        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}
        return cls(**kwargs)

    def __str__(self) -> str:
        return self.data.__str__()

    def __getitem__(self, item) -> Self | pandas.Series:
        """
        Access a column, row, or filtered subset of the OC data.

        Parameters
        ----------
        item : int, str, slice, or array-like
            Specifies which part of the OC data to retrieve:
            - `int` : returns a new `OC` instance containing a single row.
            - `str` : returns the column with the given name as a `pandas.Series`.
            - `slice` or array-like : returns a new `OC` instance with the selected rows.

        Returns
        -------
        OC or pandas.Series
            - `OC` instance for row or filtered selection.
            - `pandas.Series` for column access.

        Examples
        --------
        Access a column:

        >>> oc['oc']
        0    0.10
        1   -0.05
        2    0.00
        Name: oc, dtype: float64

        Access a single row:

        >>> row = oc[0]
        >>> isinstance(row, OC)
        True
        >>> row.data
           minimum_time  minimum_time_error  weights  minimum_type labels  cycle    oc
        0   2450000.0               0.01     100          None  None    0.0  0.10

        Access multiple rows using a slice:

        >>> subset = oc[0:2]
        >>> len(subset)
        2
        """
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
        """
        Assign values to a column in the OC data.

        Parameters
        ----------
        key : str
            Name of the column to assign or update.
        value : array-like or scalar
            Values to set for the column. Length should match the number of rows
            in the OC data; scalars are broadcast to all rows.

        Returns
        -------
        None

        Notes
        -----
        - If the column does not exist, it will be created.
        - If the column exists, its values are overwritten.
        - This method modifies the internal `data` DataFrame in-place.

        Examples
        --------
        Assign a new column:

        >>> oc['new_col'] = [1, 2, 3]

        Update an existing column:

        >>> oc['weights'] = [50, 50, 50]
        """
        self.data.loc[:, key] = value

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _equal_bins(df: pd.DataFrame, xcol: str, bin_count: int) -> np.ndarray:
        """
        Compute equal-width bin edges for a given column in the data.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data to be binned.
        xcol : str
            Name of the column to use as the x-axis for binning.
        bin_count : int
            Number of bins to create.

        Returns
        -------
        np.ndarray
            Array of shape `(bin_count + 1,)` containing the bin edges.

        Notes
        -----
        - The bins are linearly spaced between the minimum and maximum of `xcol`.
        - This method only returns edges; bin assignment is handled elsewhere.

        Examples
        --------
        >>> df = pd.DataFrame({'cycle': [0, 1, 2, 3, 4, 5]})
        >>> OC._equal_bins(df, 'cycle', 3)
        array([0., 1.6667, 3.3333, 5.])
        """
        xvals = df[xcol].to_numpy(dtype=float)
        xmin, xmax = xvals.min(), xvals.max()
        edges = np.linspace(xmin, xmax, bin_count + 1)
        return edges

    @staticmethod
    def _smart_bins(
            df: pd.DataFrame,
            xcol: str,
            bin_count: int,
            smart_bin_period: float = 50.0
    ) -> np.ndarray:
        """
        Compute bins for a column using adaptive "smart" binning to handle gaps.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the data to be binned.
        xcol : str
            Name of the column to use as the x-axis for binning.
        bin_count : int
            Target number of bins to create.
        smart_bin_period : float, optional
            Threshold for detecting large gaps in the data. Default is 50.0.

        Returns
        -------
        np.ndarray
            Array of shape `(N, 2)` containing start and end edges for each bin.

        Raises
        ------
        ValueError
            If `smart_bin_period` is not a positive number.

        Notes
        -----
        - The algorithm first detects large gaps greater than `smart_bin_period`
          to split bins, then merges or subdivides bins to match `bin_count`.
        - Useful for unevenly spaced data to avoid empty or overly sparse bins.
        - Output is a 2D array with each row `[start, end]` defining a bin interval.

        Examples
        --------
        >>> df = pd.DataFrame({'cycle': [0, 1, 2, 100, 101, 102]})
        >>> OC._smart_bins(df, 'cycle', bin_count=3, smart_bin_period=50)
        array([[  0.,   2.],
               [100., 102.]])
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
            bin_method: Optional[ArrayReducer] = None,
            bin_error_method: Optional[ArrayReducer] = None,
            bin_style: Optional[Callable[[pd.DataFrame, int], np.ndarray]] = None
    ) -> Self:
        """
        Bin the O–C data along the cycle axis and compute weighted averages and errors.

        Parameters
        ----------
        bin_count : int, optional
            Number of bins to create. Default is 1.
        bin_method : callable, optional
            Function to compute binned values from array and weights:
            `func(array: np.ndarray, weights: np.ndarray) -> float`.
            Default is weighted mean.
        bin_error_method : callable, optional
            Function to compute binned errors from weights: `func(weights: np.ndarray) -> float`.
            Default is `1 / sqrt(sum(weights))`.
        bin_style : callable, optional
            Function to define custom bin edges:
            `func(df: pd.DataFrame, bin_count: int) -> np.ndarray of shape (N, 2)`.

        Returns
        -------
        OC
            New OC instance containing binned data.

        Raises
        ------
        ValueError
            If `cycle`, `oc`, or `weights` columns are missing or contain NaNs.
            If `bin_count` is not positive.

        Notes
        -----
        - Uses `cycle` as the x-axis and `oc` as the y-axis.
        - By default, bins are equal-width and weighted averages are used.
        - Supports custom binning logic via `bin_style`.
        - Binned rows have:
            - `labels` set to "Binned"
            - `minimum_time` and `weights` set to NaN
            - `minimum_type` set to None

        Examples
        --------
        Bin the data into 5 equal-width bins:

        >>> binned_oc = oc.bin(bin_count=5)

        Bin using a custom error function and custom bin edges:

        >>> def custom_bin_style(df, n_bins):
        ...     return np.array([[0, 2], [2, 5], [5, 10]])
        >>> binned_oc = oc.bin(bin_count=3, bin_error_method=lambda w: np.sqrt(np.sum(w)), bin_style=custom_bin_style)
        """
        if "cycle" in self.data.columns:
            xcol = "cycle"
        else:
            raise ValueError("`OC.bin` needs or 'cycle' column as x-axis.")

        if "oc" not in self.data.columns:
            raise ValueError("`oc` column is required")

        if "weights" not in self.data.columns:
            raise ValueError("`weights` column is required")

        if self.data["weights"].hasnans:
            raise ValueError("`weights` contain NaN values")

        if self.data[xcol].hasnans:
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
            # The _equal_bins now returns edges, not bins (start, end pairs).
            # The binning logic below expects bins (start, end pairs).
            # I will convert edges to bins for compatibility with the existing loop.
            edges = self._equal_bins(self.data, xcol, int(bin_count))
            bins = np.column_stack([edges[:-1], edges[1:]])
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
            cycle=new_df["cycle"].tolist(),
            oc=new_df["oc"].tolist(),
        )

    def merge(self, oc: Self) -> Self:
        """
        Merge another `OC` instance into the current one.

        Parameters
        ----------
        oc : OC
            Another OC instance whose data will be appended.

        Returns
        -------
        OC
            A new OC instance containing the combined data of both instances.

        Notes
        -----
        - The merge is performed row-wise, preserving all columns.
        - Indexes are reset in the resulting DataFrame.
        - The original OC instances are not modified; the merge returns a new instance.

        Examples
        --------
        Merge two OC datasets:

        >>> oc1 = OC.from_file("observations1.csv")
        >>> oc2 = OC.from_file("observations2.csv")
        >>> merged = oc1.merge(oc2)
        >>> len(merged) == len(oc1) + len(oc2)
        True
        """
        new_oc = deepcopy(self)
        new_oc.data = pd.concat([self.data, oc.data], ignore_index=True, sort=False)
        return new_oc

    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit") -> Self:
        """
        Compute O–C (Observed minus Calculated) values based on a reference ephemeris.

        Parameters
        ----------
        reference_minimum : float
            Reference time of a primary minimum (e.g., first observed eclipse).
        reference_period : float
            Reference period for the cycle calculation.
        model_type : str, optional
            Type of model to return:
            - `"lmfit"` : returns an `OCLMFit` instance if available.
            - other : returns an `OC` instance.
            Default is `"lmfit"`.

        Returns
        -------
        OC
            New OC (or OCLMFit) instance with calculated `cycle` and `oc` columns.

        Raises
        ------
        ValueError
            If the `minimum_time` column is missing.
            If `minimum_type` contains unrecognized values for secondary/primary distinction.

        Notes
        -----
        - The method computes the cycle number as `(t - reference_minimum) / reference_period`,
          rounded to the nearest integer.
        - If `minimum_type` is present, secondary minima (e.g., "II", "sec") are offset by 0.5 cycles.
        - Calculated O–C values are added as the `oc` column; cycle numbers as `cycle`.
        - Other columns (`minimum_time_error`, `weights`, `labels`, etc.) are preserved.

        Examples
        --------
        Compute O–C for an OC dataset:

        >>> oc = OC.from_file("observations.csv")
        >>> oc_calc = oc.calculate_oc(reference_minimum=2450000.0, reference_period=1.2345)
        >>> oc_calc.data[['cycle', 'oc']].head()
           cycle     oc
        0    0.0  0.012
        1    1.0 -0.005
        2    2.0  0.003
        """
        df = self.data.copy()
        if "minimum_time" not in df.columns:
            raise ValueError("`minimum_time` column is required to compute O–C.")

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

        new_data: Dict[str, Optional[list]] = {
            "minimum_time": df["minimum_time"].tolist(),
            "minimum_time_error": df["minimum_time_error"].tolist() if "minimum_time_error" in df else None,
            "weights": df["weights"].tolist() if "weights" in df else None,
            "minimum_type": df["minimum_type"].tolist() if "minimum_type" in df else None,
            "labels": df["labels"].tolist() if "labels" in df else None,
        }

        if model_type == "lmfit":
            try:
                from .oc_lmfit import OCLMFit
                Target = OCLMFit
            except Exception:
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

    def residue(self, coefficients: ModelResult) -> Self:
        """
        Compute the residuals of the O–C data relative to a fitted model.

        Parameters
        ----------
        coefficients : ModelResult
            Fitted model object (from `lmfit`) containing the optimized parameters
            used to compute model-predicted O–C values.

        Returns
        -------
        OC
            New OC instance containing the residuals (`observed - model`) in the `oc` column.
            Other columns are preserved from the original instance.

        Notes
        -----
        - This method does not modify the original OC instance; a new instance is returned.
        - Residuals are computed using the model defined by `coefficients` applied to the `cycle` column.
        - Typically used after `OC.fit` or similar model fitting methods.

        Examples
        --------
        Compute residuals after fitting a linear model:

        >>> result = oc.fit_linear(a=1.0, b=0.0)  # returns a ModelResult
        >>> oc_resid = oc.residue(result)
        >>> oc_resid.data[['cycle', 'oc']].head()
           cycle       oc
        0    0.0   0.002
        1    1.0  -0.001
        2    2.0   0.003
        """
        pass

    def fit(self, functions: Union[List[ModelComponentModel], ModelComponentModel]) -> ModelResult:
        """
       Fit one or more model components to the O–C data.

       Parameters
       ----------
       functions : ModelComponentModel or list of ModelComponentModel
           Single or multiple model components (e.g., Linear, Quadratic, Sinusoidal, Keplerian)
           to fit to the `oc` data. Each component defines its own functional form and parameters.

       Returns
       -------
       ModelResult
           Fitted model result (from `lmfit`) containing optimized parameters,
           fit statistics, and the model-predicted O–C values.

       Raises
       ------
       ValueError
           If the required columns (`cycle`, `oc`, or `weights`) are missing or contain NaNs.
       TypeError
           If `functions` is not a `ModelComponentModel` or list thereof.

       Notes
       -----
       - Supports combining multiple model components into a single composite fit.
       - Uses weighted fitting based on the `weights` column.
       - The resulting `ModelResult` can be used to evaluate residuals or plot the fitted model.
       - Internally, converts `functions` to a composite `lmfit` model if multiple components are provided.

       Examples
       --------
       Fit a linear and sinusoidal model to O–C data:

       >>> linear = Linear(a=0.0, b=0.0)
       >>> sinus = Sinusoidal(amp=0.01, P=10.0)
       >>> result = oc.fit([linear, sinus])
       >>> result.best_values
       {'a': 0.0012, 'b': -0.0023, 'amp': 0.0098, 'P': 9.97}
       """
        pass

    def fit_keplerian(
            self,
            *,
            amp: Optional[ParameterModel] = None,
            e: Optional[ParameterModel] = None,
            omega: Optional[ParameterModel] = None,
            P: Optional[ParameterModel] = None,
            T: Optional[ParameterModel] = None
    ) -> ModelComponentModel:
        """
        Fit a Keplerian (elliptical orbit) model to the O–C data.

        Parameters
        ----------
        amp : ParameterModel, optional
            Amplitude of the Keplerian signal.
        e : ParameterModel, optional
            Orbital eccentricity.
        omega : ParameterModel, optional
            Argument of periastron in degrees.
        P : ParameterModel, optional
            Orbital period.
        T : ParameterModel, optional
            Time of periastron passage.

        Returns
        -------
        ModelComponentModel
            Fitted Keplerian model component with optimized parameters.

        Notes
        -----
        - Uses the Keplerian formula to model periodic variations in O–C data.
        - Supports optional `ParameterModel` objects for each orbital parameter; unspecified
          parameters can be estimated from the data.
        - The returned model component can be used to evaluate predicted O–C values or
          for further combination with other model components.

        Examples
        --------
        Fit a Keplerian model with initial guesses:

        >>> kepler_model = oc.fit_keplerian(
        ...     amp=Parameter(value=0.01),
        ...     e=Parameter(value=0.1),
        ...     omega=Parameter(value=90),
        ...     P=Parameter(value=1000),
        ...     T=Parameter(value=2450000)
        ... )
        >>> kepler_model.params['amp'].value
        0.0098
        """
        pass

    def fit_lite(
            self,
            *,
            amp: Optional[ParameterModel] = None,
            e: Optional[ParameterModel] = None,
            omega: Optional[ParameterModel] = None,
            P: Optional[ParameterModel] = None,
            T: Optional[ParameterModel] = None
    ) -> ModelComponentModel:
        """
        Fit a simplified Keplerian model to the O–C data.

        Parameters
        ----------
        amp : ParameterModel, optional
            Amplitude of the Keplerian signal.
        e : ParameterModel, optional
            Orbital eccentricity.
        omega : ParameterModel, optional
            Argument of periastron in degrees.
        P : ParameterModel, optional
            Orbital period.
        T : ParameterModel, optional
            Time of periastron passage.

        Returns
        -------
        ModelComponentModel
            Fitted simplified Keplerian model component with optimized parameters.

        Notes
        -----
        - This method provides a lighter or approximate version of a full Keplerian fit.
        - Useful for initial estimates or for datasets where full Keplerian fitting is unnecessary.
        - Supports optional `ParameterModel` objects for each orbital parameter; unspecified
          parameters can be estimated from the data.
        - The returned model component can be evaluated for predicted O–C values or combined
          with other model components.

        Examples
        --------
        Fit a simplified Keplerian model:

        >>> kepler_lite = oc.fit_lite(
        ...     amp=Parameter(value=0.01),
        ...     e=Parameter(value=0.05),
        ...     P=Parameter(value=1000),
        ... )
        >>> kepler_lite.params['P'].value
        1000.0
        """
        pass

    def fit_linear(
            self,
            *,
            a: Optional[ParameterModel] = None,
            b: Optional[ParameterModel] = None
    ) -> ModelComponentModel:
        """
        Fit a linear model to the O–C data.

        Parameters
        ----------
        a : ParameterModel, optional
            Slope of the linear model. If not provided, it will be estimated from the data.
        b : ParameterModel, optional
            Intercept of the linear model. If not provided, it will be estimated from the data.

        Returns
        -------
        ModelComponentModel
            Fitted linear model component with optimized parameters.

        Notes
        -----
        - The linear model has the form `oc = a * cycle + b`.
        - Optional `ParameterModel` objects can be used to fix or initialize parameters.
        - The returned model component can be used to compute predicted O–C values or
          combined with other model components for composite fitting.

        Examples
        --------
        Fit a linear trend to O–C data:

        >>> linear_model = oc.fit_linear(a=Parameter(value=0.001), b=Parameter(value=-0.002))
        >>> linear_model.params['a'].value
        0.001
        >>> linear_model.params['b'].value
        -0.002
        """
        pass

    def fit_quadratic(
            self,
            *,
            q: Optional[ParameterModel] = None
    ) -> ModelComponentModel:
        """
        Fit a quadratic model to the O–C data.

        Parameters
        ----------
        q : ParameterModel, optional
            Quadratic coefficient. If not provided, it will be estimated from the data.

        Returns
        -------
        ModelComponentModel
            Fitted quadratic model component with optimized parameter(s).

        Notes
        -----
        - The quadratic model has the form `oc = q * cycle^2`.
        - Optional `ParameterModel` can be used to fix or initialize the quadratic coefficient.
        - The returned model component can be used to compute predicted O–C values
          or combined with other model components for composite fitting.

        Examples
        --------
        Fit a quadratic trend to O–C data:

        >>> quad_model = oc.fit_quadratic(q=Parameter(value=0.0001))
        >>> quad_model.params['q'].value
        0.0001
        """
        pass

    def fit_sinusoidal(
            self,
            *,
            amp: Optional[ParameterModel] = None,
            P: Optional[ParameterModel] = None
    ) -> ModelComponentModel:
        """
        Fit a sinusoidal model to the O–C data.

        Parameters
        ----------
        amp : ParameterModel, optional
            Amplitude of the sinusoidal signal. If not provided, it will be estimated from the data.
        P : ParameterModel, optional
            Period of the sinusoidal signal. If not provided, it will be estimated from the data.

        Returns
        -------
        ModelComponentModel
            Fitted sinusoidal model component with optimized parameters.

        Notes
        -----
        - The sinusoidal model has the form `oc = amp * sin(2 * pi * cycle / P)`.
        - Optional `ParameterModel` objects can be used to fix or initialize parameters.
        - The returned model component can be evaluated for predicted O–C values
          or combined with other model components in a composite fit.

        Examples
        --------
        Fit a sinusoidal variation to O–C data:

        >>> sinus_model = oc.fit_sinusoidal(amp=Parameter(value=0.01), P=Parameter(value=1000))
        >>> sinus_model.params['amp'].value
        0.01
        >>> sinus_model.params['P'].value
        1000
        """
        pass

    def fit_parabola(
            self,
            *,
            q: Optional[ParameterModel] = None,
            a: Optional[ParameterModel] = None,
            b: Optional[ParameterModel] = None
    ) -> ModelComponentModel:
        """
        Fit a parabolic model to the O–C data.

        Parameters
        ----------
        q : ParameterModel, optional
            Quadratic coefficient. If not provided, it will be estimated from the data.
        a : ParameterModel, optional
            Linear coefficient. If not provided, it will be estimated from the data.
        b : ParameterModel, optional
            Constant term (intercept). If not provided, it will be estimated from the data.

        Returns
        -------
        ModelComponentModel
            Fitted parabolic model component with optimized parameters.

        Notes
        -----
        - The parabolic model has the form `oc = q * cycle^2 + a * cycle + b`.
        - Optional `ParameterModel` objects can be used to fix or initialize coefficients.
        - The returned model component can be used to compute predicted O–C values
          or combined with other model components for composite fitting.

        Examples
        --------
        Fit a parabolic trend to O–C data:

        >>> parab_model = oc.fit_parabola(q=Parameter(value=0.0001),
        ...                               a=Parameter(value=0.001),
        ...                               b=Parameter(value=-0.002))
        >>> parab_model.params['q'].value
        0.0001
        >>> parab_model.params['a'].value
        0.001
        >>> parab_model.params['b'].value
        -0.002
        """
        pass

    def plot(
            self,
            model: Union[InferenceData, ModelResult, List[ModelComponent]] = None,
            *,
            ax: Optional["plt.Axes"] = None,
            res_ax: Optional["plt.Axes"] = None,
            res: bool = True,
            title: Optional[str] = None,
            x_col: str = "cycle",
            y_col: str = "oc",
            fig_size: tuple = (10, 7),
            plot_kwargs: Optional[dict] = None,
            extension_factor: float = 0.05,
            model_components: Optional[list] = None
    ) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
        """
        Plot the O–C data along with optional model predictions and residuals.

        Parameters
        ----------
        model : InferenceData, ModelResult, or list of ModelComponent, optional
            Model or model components to overlay on the O–C plot.
        ax : matplotlib.axes.Axes, optional
            Axes object for the main plot. If None, a new figure and axes are created.
        res_ax : matplotlib.axes.Axes, optional
            Axes object for residuals plot. Only used if `res=True`.
        res : bool, default=True
            If True, show residuals between the data and model on a separate subplot.
        title : str, optional
            Title for the plot.
        x_col : str, default="cycle"
            Column name in the data to use for the x-axis.
        y_col : str, default="oc"
            Column name in the data to use for the y-axis.
        fig_size : tuple, default=(10, 7)
            Figure size if a new figure is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to the plotting function (e.g., marker, linestyle, color).
        extension_factor : float, default=0.05
            Fractional extension for axis limits to improve visual spacing.
        model_components : list, optional
            List of individual model components to display separately on the plot.

        Returns
        -------
        matplotlib.axes.Axes or dict
            If `res=False`, returns the main Axes object. If `res=True`, returns a dictionary
            with keys `'main_ax'` and `'res_ax'` containing the corresponding Axes objects.

        Notes
        -----
        - This method uses the `Plot.plot` function from the visualization module.
        - Supports overlaying multiple model components or posterior predictive samples.
        - Automatically handles binning or weighting if present in the data.
        - Residuals plot shows data minus model values for quick assessment of fit quality.

        Examples
        --------
        Plot O–C data with a fitted sinusoidal model:

        >>> sinus_model = oc.fit_sinusoidal(amp=Parameter(value=0.01), P=Parameter(value=1000))
        >>> oc.plot(model=sinus_model, title="O–C with Sinusoidal Fit")

        Plot data with residuals on a separate subplot:

        >>> oc.plot(model=sinus_model, res=True, fig_size=(12, 8))
        """
        from .visualization import Plot
        return Plot.plot(
            self,
            model=model,
            ax=ax,
            res_ax=res_ax,
            res=res,
            title=title,
            x_col=x_col,
            y_col=y_col,
            fig_size=fig_size,
            plot_kwargs=plot_kwargs,
            extension_factor=extension_factor,
            model_components=model_components,
        )
