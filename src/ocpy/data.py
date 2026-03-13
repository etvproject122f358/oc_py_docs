from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Callable

from typing_extensions import Self

import pandas as pd
import numpy as np
from copy import deepcopy

from ocpy.model_data import DataModel
from ocpy.custom_types import BinarySeq
from ocpy.utils import Fixer

from .errors import LengthCheckError
from .oc import OC
from .oc_lmfit import OCLMFit
from .oc_pymc import OCPyMC


class Data(DataModel):
    """
    Container for eclipse minimum timing data.

    The `Data` class stores and manages observational times of minima
    (e.g., eclipse timings of binary stars) together with optional
    metadata such as timing uncertainties, observational weights,
    minimum type (primary/secondary), and labels.

    Internally the data are stored in a :class:`pandas.DataFrame`
    with standardized column names. The class provides utilities
    for safely manipulating the dataset, including:

    * Filling or computing timing uncertainties and weights
    * Loading data from files
    * Computing O–C (Observed minus Calculated) values
    * Grouping or merging datasets

    The class is designed to behave like a lightweight table object
    while preserving domain-specific semantics required for
    O–C analysis of eclipsing binaries.

    Parameters
    ----------
    minimum_time : list-like
        Times of observed minima (typically in Julian Date or BJD).
        This field is required.
    minimum_time_error : list-like, optional
        Uncertainties associated with each minimum time. If provided,
        the length must match ``minimum_time``.
    weights : list-like, optional
        Weights assigned to each observation.
    minimum_type : BinarySeq, optional
        Indicator of minimum type (e.g., primary or secondary eclipse).
        Accepted representations may include integers, strings
        (e.g., ``"I"``, ``"II"``, ``"primary"``, ``"secondary"``), or
        other binary encodings.
    labels : list-like, optional
        Optional labels or identifiers for each observation.

    Raises
    ------
    ValueError
        If ``minimum_time`` is ``None``.
    LengthCheckError
        If provided sequences do not match the length of
        ``minimum_time``.

    Notes
    -----
    All input sequences are normalized internally so that their
    lengths match the length of ``minimum_time``. Missing values
    are automatically expanded or filled using utilities from
    :mod:`ocpy.utils`.

    The underlying storage is a :class:`pandas.DataFrame` with
    the following standard columns:

    - ``minimum_time``
    - ``minimum_time_error``
    - ``weights``
    - ``minimum_type``
    - ``labels``

    Most modification methods return a **new `Data` instance**
    rather than mutating the existing object.

    Examples
    --------
    Create a dataset with minimum times:

    >>> from ocpy import Data
    >>> d = Data(minimum_time=[2450000.1, 2450001.3, 2450002.5])

    With uncertainties and types:

    >>> d = Data(
    ...     minimum_time=[2450000.1, 2450001.3],
    ...     minimum_time_error=[0.0002, 0.0003],
    ...     minimum_type=["I", "II"]
    ... )

    Access the underlying table:

    >>> d.data
    """

    def __init__(
            self,
            minimum_time: List,
            minimum_time_error: Optional[List] = None,
            weights: Optional[List] = None,
            minimum_type: Optional[BinarySeq] = None,
            labels: Optional[List] = None
    ) -> None:
        """
        Initialize a `Data` object containing eclipse minimum timing data.

        The constructor creates a standardized internal
        :class:`pandas.DataFrame` containing the provided timing
        measurements and associated metadata. All optional sequences
        are automatically adjusted to match the length of
        ``minimum_time``.

        Parameters
        ----------
        minimum_time : list-like
            Times of observed minima. These are typically given in
            Julian Date (JD), Barycentric Julian Date (BJD), or a
            similar astronomical time system. This parameter is required
            and defines the length of the dataset.
        minimum_time_error : list-like or float, optional
            Uncertainties associated with each minimum time. If a scalar
            value is provided, it will be broadcast to all observations.
            If a sequence is provided, its length must match
            ``minimum_time``.
        weights : list-like or float, optional
            Weights assigned to each observation. If provided as a scalar,
            it will be applied to all observations.
        minimum_type : BinarySeq, optional
            Indicator of the type of minimum. This is commonly used to
            distinguish between primary and secondary eclipses.
            Typical values include ``"I"``, ``"II"``, ``"primary"``,
            ``"secondary"``, or numeric equivalents.
        labels : list-like, optional
            Optional labels or identifiers for each observation,
            such as instrument names, observers, or literature sources.

        Raises
        ------
        ValueError
            If ``minimum_time`` is ``None``.

        Notes
        -----
        All optional sequences are normalized using
        :func:`ocpy.utils.Fixer.length_fixer` so that their lengths
        match the length of ``minimum_time``. Scalars are automatically
        expanded to the correct size.

        Internally, the data are stored in a :class:`pandas.DataFrame`
        with the following columns:

        - ``minimum_time``
        - ``minimum_time_error``
        - ``weights``
        - ``minimum_type``
        - ``labels``

        Examples
        --------
        Create a dataset with only minimum times:

        >>> Data(minimum_time=[2450000.1, 2450001.2, 2450002.4])

        Provide uncertainties and eclipse types:

        >>> Data(
        ...     minimum_time=[2450000.1, 2450001.2],
        ...     minimum_time_error=[0.0002, 0.0003],
        ...     minimum_type=["I", "II"]
        ... )

        Use a scalar uncertainty applied to all observations:

        >>> Data(
        ...     minimum_time=[2450000.1, 2450001.2],
        ...     minimum_time_error=0.0002
        ... )
        """
        if minimum_time is None:
            raise ValueError("`minimum_time` is required and cannot be None.")

        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)

        # Convert to list if it's a scalar/None to avoid pandas scalar error
        minimum_time_sequence = minimum_time if hasattr(minimum_time, "__len__") else [minimum_time]

        self.data = pd.DataFrame(
            {
                "minimum_time": minimum_time_sequence,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
            }
        )

    def __str__(self) -> str:
        return self.data.__str__()

    def __getitem__(self, item) -> Union[Self, pd.Series]:
        """
        Retrieve data from the dataset.

        This method provides flexible indexing behavior similar to a
        :class:`pandas.DataFrame`. Depending on the type of ``item``,
        it can return either a column, a single-row `Data` object, or
        a filtered `Data` object.

        Parameters
        ----------
        item : str, int, slice, array-like, or pandas-compatible indexer
            Index or key used to access the data.

            * ``str`` – Returns the corresponding column as a
              :class:`pandas.Series`.
            * ``int`` – Returns a new `Data` object containing only
              the selected row.
            * Other indexers (e.g., slices, boolean masks, lists) –
              Returns a new `Data` object containing the filtered rows.

        Returns
        -------
        Data or pandas.Series
            - If ``item`` is a string, the corresponding column is returned
              as a :class:`pandas.Series`.
            - Otherwise, a new `Data` object containing the selected rows
              is returned.

        Notes
        -----
        This method intentionally mimics the behavior of
        :class:`pandas.DataFrame` indexing while preserving the `Data`
        abstraction.

        When row selection is performed, the result is wrapped into a
        new `Data` instance to ensure that all domain-specific methods
        (e.g., O–C calculations) remain available.

        Examples
        --------
        Select a column:

        >>> d["minimum_time"]

        Select a single observation:

        >>> d[0]

        Filter rows using a boolean mask:

        >>> mask = d["weights"] > 1
        >>> d_filtered = d[mask]

        Slice the dataset:

        >>> d_subset = d[:5]
        """
        if isinstance(item, str):

            return self.data[item]
        elif isinstance(item, int):
            row = self.data.iloc[item]
            return Data(
                minimum_time=[row["minimum_time"]],
                minimum_time_error=[row["minimum_time_error"]],
                weights=[row["weights"]],
                minimum_type=[row["minimum_type"]],
                labels=[row["labels"]],
            )
        else:
            filtered_table = self.data[item]

            return Data(
                minimum_time=filtered_table["minimum_time"],
                minimum_time_error=filtered_table["minimum_time_error"],
                weights=filtered_table["weights"],
                minimum_type=filtered_table["minimum_type"],
                labels=filtered_table["labels"],
            )

    def __setitem__(self, key, value) -> None:
        """
        Assign values to a column in the dataset.

        This method allows column assignment using dictionary-like
        syntax, similar to :class:`pandas.DataFrame`. The specified
        column will be created if it does not already exist, or
        overwritten if it does.

        Parameters
        ----------
        key : str
            Name of the column to assign or modify.
        value : scalar or array-like
            Values to assign to the column. Scalars will be broadcast
            to all rows, while array-like objects must have a length
            compatible with the dataset.

        Returns
        -------
        None

        Notes
        -----
        This operation modifies the underlying :class:`pandas.DataFrame`
        in place.

        Unlike most other methods of `Data`, which return a new instance,
        ``__setitem__`` directly mutates the existing object.

        Examples
        --------
        Add a new column:

        >>> d["observer"] = ["A", "B", "C"]

        Assign a scalar value to all rows:

        >>> d["instrument"] = "TESS"

        Modify an existing column:

        >>> d["weights"] = [1.0, 0.5, 2.0]
        """
        self.data.loc[:, key] = value

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_file(cls, file: Union[str, Path], columns: Optional[Dict[str, str]] = None) -> Self:
        """
        Create a `Data` object from a tabular file.

        This method reads timing data from a CSV or Excel file and converts
        it into a `Data` instance. Column names in the file can optionally be
        mapped to the standardized column names used by the `Data` class.

        Parameters
        ----------
        file : str or pathlib.Path
            Path to the input file. Supported formats are:

            - ``.csv``
            - ``.xls``
            - ``.xlsx``

        columns : dict of str to str, optional
            Mapping between the standardized `Data` column names and the
            column names present in the file.

            Two mapping styles are accepted:

            1. **Standard → file column name**

               Example:

               >>> columns = {"minimum_time": "BJD", "minimum_time_error": "err"}

            2. **File column name → standard**

               Example:

               >>> columns = {"BJD": "minimum_time", "err": "minimum_time_error"}

            The standard column names recognized by the `Data` class are:

            - ``minimum_time``
            - ``minimum_time_error``
            - ``weights``
            - ``minimum_type``
            - ``labels``

        Returns
        -------
        Data
            A new `Data` instance containing the imported observations.

        Raises
        ------
        ValueError
            If the file format is not supported.

        ValueError
            If the file does not contain a column corresponding to
            ``minimum_time``.

        Notes
        -----
        The ``minimum_time`` column is required to construct a valid
        dataset. All other columns are optional and will be set to
        ``None`` if not present in the file.

        Internally, the file is loaded using :func:`pandas.read_csv`
        or :func:`pandas.read_excel`.

        Examples
        --------
        Load a dataset from a CSV file:

        >>> d = Data.from_file("minima.csv")

        Load a file with custom column names:

        >>> d = Data.from_file(
        ...     "observations.csv",
        ...     columns={"BJD": "minimum_time", "err": "minimum_time_error"}
        ... )

        Load an Excel file:

        >>> d = Data.from_file("minima.xlsx")
        """
        file_path = Path(file)

        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Use `csv`, `xls`, or `xlsx` instead")

        expected = ["minimum_time", "minimum_time_error", "weights", "minimum_type", "labels"]
        if columns:
            if any(k in expected for k in columns.keys()):
                rename_map = {v: k for k, v in columns.items()}
            else:
                rename_map = columns
            df = df.rename(columns=rename_map)

        if "minimum_time" not in df.columns:
            available = list(df.columns)
            raise ValueError(f"Could not find 'minimum_time' in file columns. Available columns: {available}. "
                             f"Please check your 'columns' mapping.")

        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}

        return cls(**kwargs)

    def _assign_or_fill(self, df: pd.DataFrame, col: str, values, override: bool) -> None:
        """
        Assign values to a DataFrame column or fill missing entries.

        If ``override`` is ``True`` or the column does not exist in the
        DataFrame, the column is assigned directly with the provided
        values. Otherwise, only the entries that are ``NaN`` are replaced,
        leaving existing non-null values unchanged.

        Parameters
        ----------
        df : pandas.DataFrame
            Target DataFrame in which the column will be modified.
        col : str
            Name of the column to assign or update.
        values : scalar or array-like
            Values to assign to the column. Scalars may be broadcast
            by pandas, while array-like values must be compatible with
            the DataFrame length.
        override : bool
            If ``True``, the column is completely replaced with the
            provided values. If ``False``, only missing values
            (``NaN``) in the existing column are filled.

        Returns
        -------
        None

        Notes
        -----
        This is an internal utility method used by functions such as
        :meth:`fill_errors`, :meth:`fill_weights`, and
        :meth:`calculate_weights` to ensure consistent column updates.
        The operation modifies the provided DataFrame in place.
        """
        if override or col not in df.columns:
            df[col] = values
        else:
            base = df[col]
            df[col] = base.where(~pd.isna(base), values)

    def fill_errors(self, errors: Union[List, Tuple, np.ndarray, float], override: bool = False) -> Self:
        """
        Fill or assign timing uncertainties for the dataset.

        This method returns a new `Data` object in which the
        ``minimum_time_error`` column is populated using the provided
        values. Existing values can optionally be preserved or replaced.

        Parameters
        ----------
        errors : list, tuple, numpy.ndarray, or float
            Uncertainty values to assign to ``minimum_time_error``.

            - If a scalar is provided, it will be applied to all rows.
            - If an array-like object is provided, its length must match
              the number of observations in the dataset.
        override : bool, default=False
            If ``True``, all existing values in the ``minimum_time_error``
            column are replaced.

            If ``False``, only entries that are currently ``NaN`` are filled,
            leaving existing values unchanged.

        Returns
        -------
        Data
            A new `Data` instance with updated ``minimum_time_error`` values.

        Raises
        ------
        LengthCheckError
            If ``errors`` is array-like and its length does not match the
            number of rows in the dataset.

        Notes
        -----
        This method does not modify the original object. Instead,
        it returns a new `Data` instance with the updated values.

        Internally, column assignment is handled by the private
        :meth:`_assign_or_fill` method to ensure consistent behavior
        across similar operations.

        Examples
        --------
        Assign a constant uncertainty to all observations:

        >>> d2 = d.fill_errors(0.0002)

        Fill only missing uncertainties:

        >>> d2 = d.fill_errors([0.0002, 0.0003, 0.00025])

        Replace all existing uncertainties:

        >>> d2 = d.fill_errors(0.0002, override=True)
        """
        new_data = deepcopy(self)
        if isinstance(errors, (list, tuple, np.ndarray)) and len(errors) != len(new_data.data):
            raise LengthCheckError("Length of `errors` must be equal to the length of the data")
        self._assign_or_fill(new_data.data, "minimum_time_error", errors, override)
        return new_data

    def fill_weights(self, weights: Union[List, Tuple, np.ndarray, float], override: bool = False) -> Self:
        """
        Fill or assign observational weights for the dataset.

        This method returns a new `Data` object in which the
        ``weights`` column is populated using the provided values.
        Existing weights can optionally be preserved or replaced.

        Parameters
        ----------
        weights : list, tuple, numpy.ndarray, or float
            Weight values to assign to the ``weights`` column.

            - If a scalar is provided, it will be applied to all rows.
            - If an array-like object is provided, its length must match
              the number of observations in the dataset.
        override : bool, default=False
            If ``True``, all existing values in the ``weights`` column
            are replaced.

            If ``False``, only entries that are currently ``NaN`` are filled,
            leaving existing values unchanged.

        Returns
        -------
        Data
            A new `Data` instance with updated ``weights`` values.

        Raises
        ------
        LengthCheckError
            If ``weights`` is array-like and its length does not match the
            number of rows in the dataset.

        Notes
        -----
        This method does not modify the original object. Instead,
        it returns a new `Data` instance with the updated values.

        Internally, column assignment is handled by the private
        :meth:`_assign_or_fill` method to ensure consistent behavior
        across similar operations.

        Examples
        --------
        Assign a constant weight to all observations:

        >>> d2 = d.fill_weights(1.0)

        Fill only missing weights:

        >>> d2 = d.fill_weights([1.0, 0.5, 2.0])

        Replace all existing weights:

        >>> d2 = d.fill_weights(1.0, override=True)
        """
        new_data = deepcopy(self)
        if isinstance(weights, (list, tuple, np.ndarray)) and len(weights) != len(new_data.data):
            raise LengthCheckError("Length of `weights` must be equal to the length of the data")
        self._assign_or_fill(new_data.data, "weights", weights, override)
        return new_data

    def calculate_weights(self, method: Callable[[pd.Series], pd.Series] = None, override: bool = True) -> Self:
        r"""
        Calculate observational weights based on timing uncertainties.

        This method computes weights for each observation, typically
        using the inverse-variance method, and returns a new `Data`
        instance with the updated ``weights`` column.

        Parameters
        ----------
        method : callable, optional
            A custom function to compute weights from the
            ``minimum_time_error`` series. It must accept a
            :class:`pandas.Series` of errors and return a
            :class:`pandas.Series` of weights.

            If ``None`` (default), the inverse-variance method is used:

            .. math::
                w_i = \frac{1}{\sigma_i^2}

            where :math:`\sigma_i` is the timing uncertainty
            for the i-th observation.
        override : bool, default=True
            If ``True``, existing ``weights`` values are replaced.
            If ``False``, only missing entries (``NaN``) are filled.

        Returns
        -------
        Data
            A new `Data` instance with updated weights.

        Raises
        ------
        ValueError
            If ``minimum_time_error`` contains ``NaN`` values.
        ValueError
            If ``minimum_time_error`` contains zero values, which
            would cause division by zero in the default method.
        TypeError
            If ``method`` is provided but is not callable.

        Notes
        -----
        - The default inverse-variance weighting gives higher weight
          to observations with smaller uncertainties.
        - This method does not modify the original `Data` instance;
          it returns a new instance with updated weights.
        - Internally, column assignment uses :meth:`_assign_or_fill`
          to respect the ``override`` flag.

        Examples
        --------
        Compute default inverse-variance weights:

        >>> d2 = d.calculate_weights()

        Compute weights with a custom method:

        >>> def custom_weights(errors):
        ...     return 1 / errors
        >>> d2 = d.calculate_weights(method=custom_weights, override=True)

        Fill only missing weights without overwriting existing ones:

        >>> d2 = d.calculate_weights(override=False)
        """

        def inverse_variance_weights(err_days: pd.Series) -> pd.Series:
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / np.square(err_days)

        new_data = deepcopy(self)
        minimum_time_error = new_data.data["minimum_time_error"]

        if minimum_time_error.hasnans:
            raise ValueError("minimum_time_error contains NaN value(s)")
        if (minimum_time_error == 0).any():
            raise ValueError("minimum_time_error contains `0`")

        if method is not None and not callable(method):
            raise TypeError("`method` must be callable or None for inverse variance weights")

        if method is None:
            method = inverse_variance_weights

        weights = method(minimum_time_error)
        self._assign_or_fill(new_data.data, "weights", weights, override)
        return new_data

    def calculate_oc(self, reference_minimum: float, reference_period: float, model_type: str = "lmfit") -> OC:
        """
        Compute Observed minus Calculated (O–C) values for the dataset.

        This method calculates the O–C values for each observed minimum
        based on a reference minimum time and period. The O–C values
        quantify the difference between observed and predicted
        timings, which is fundamental for analyzing period variations
        in eclipsing binaries.

        Parameters
        ----------
        reference_minimum : float
            The reference time of minimum (e.g., initial epoch) used
            to compute predicted minima.
        reference_period : float
            The reference orbital period of the system. This is used
            to compute the expected timing of each cycle.
        model_type : str, default='lmfit'
            Specifies the type of O–C model to return. Supported options:

            - ``'lmfit'`` or ``'lmfit_model'`` – returns an :class:`OCLMFit` object.
            - ``'pymc'`` or ``'pymc_model'`` – returns an :class:`OCPyMC` object.
            - Any other string – returns a generic :class:`OC` object.

        Returns
        -------
        OC
            An instance of the appropriate O–C model class
            (:class:`OC`, :class:`OCLMFit`, or :class:`OCPyMC`)
            containing:

            - ``minimum_time`` – observed minima
            - ``cycle`` – computed cycle numbers (integer or half-integer for secondary minima)
            - ``oc`` – O–C values
            - Additional columns from the original `Data` (errors, weights, labels, minimum_type)

        Raises
        ------
        ValueError
            If the ``minimum_time`` column is missing.
        Notes
        -----
        - **Cycle calculation:** The phase of each observation is computed as:

          .. math::
              \text{phase} = \frac{t - \text{reference_minimum}}{\text{reference_period}}

          The cycle number is the nearest integer to the phase.
        - **Secondary minima:** If ``minimum_type`` is present and indicates
          a secondary eclipse (e.g., "II", "secondary", 2), the cycle is
          adjusted to half-integer values:

          .. math::
              \text{cycle}_{\text{sec}} = \text{round}(\text{phase} - 0.5) + 0.5

        - **O–C computation:** The O–C value for each observation is:

          .. math::
              \text{O–C} = t_{\text{obs}} - ( \text{reference_minimum} + \text{cycle} \times \text{reference_period} )

        - The method **does not modify the original `Data`**. The returned
          object contains a copy of the original data along with the computed
          ``cycle`` and ``oc`` arrays.
        - The ``model_type`` determines which class is instantiated for
          further modeling of the O–C diagram.

        Examples
        --------
        Compute O–C values using the default LMFit model:

        >>> oc_model = d.calculate_oc(reference_minimum=2450000.0, reference_period=1.2345)

        Compute O–C values using a PyMC model:

        >>> oc_model = d.calculate_oc(
        ...     reference_minimum=2450000.0,
        ...     reference_period=1.2345,
        ...     model_type="pymc"
        ... )

        Access the computed O–C values:

        >>> oc_model.oc
        [0.0001, -0.0002, 0.0003, ...]
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

        common_kwargs = dict(
            minimum_time=new_data["minimum_time"],
            minimum_time_error=new_data["minimum_time_error"],
            weights=new_data["weights"],
            minimum_type=new_data["minimum_type"],
            labels=new_data["labels"],
            cycle=cycle,
            oc=oc,
        )

        targets = str(model_type).strip().lower()
        if targets in {"lmfit", "lmfit_model"}:
            Target = OCLMFit
        elif targets in {"pymc", "pymc_model"}:
            Target = OCPyMC
        else:
            Target = OC

        return Target(**common_kwargs)

    def merge(self, data: Self) -> Self:
        """
        Merge the current dataset with another `Data` object.

        This method concatenates the rows of the current `Data` instance
        with those of another `Data` object, returning a new `Data`
        instance. Column alignment is based on column names.

        Parameters
        ----------
        data : Data
            Another `Data` instance to merge with the current dataset.

        Returns
        -------
        Data
            A new `Data` instance containing all rows from both datasets.

        Notes
        -----
        - The original datasets are not modified.
        - Missing columns in either dataset will result in ``NaN``
          values in the merged dataset, following pandas' concatenation rules.
        - Indexes are reset in the merged dataset for consistency.

        Examples
        --------
        >>> d1 = Data(minimum_time=[2450000.1, 2450001.2])
        >>> d2 = Data(minimum_time=[2450002.3, 2450003.4])
        >>> d_merged = d1.merge(d2)
        >>> len(d_merged)
        4
        """
        new_data = deepcopy(self)
        new_data.data = pd.concat([self.data, data.data], ignore_index=True, sort=False)
        return new_data

    def group_by(self, column: str) -> List[Self]:
        """
        Split the dataset into groups based on a column.

        This method groups the `Data` object by the values in a specified
        column and returns a list of new `Data` instances, each containing
        one group of rows.

        Parameters
        ----------
        column : str
            Name of the column to group by.

        Returns
        -------
        list of Data
            A list of `Data` objects, each corresponding to one group.
            If the column is missing or contains only NaN values, a list
            with a single copy of the original dataset is returned.

        Notes
        -----
        - Grouping is performed using :meth:`pandas.DataFrame.groupby`.
        - The original `Data` object is not modified; each group is a
          deep copy.
        - NaN values are treated as a separate group unless ``dropna=True``
          in the internal pandas grouping.

        Examples
        --------
        Group a dataset by the ``minimum_type`` column:

        >>> groups = d.group_by("minimum_type")
        >>> len(groups)
        2  # e.g., one group for primary, one for secondary minima

        Access the first group:

        >>> groups[0].data
        minimum_time  minimum_type
        2450000.1    I
        2450002.3    I

        If the grouping column does not exist:

        >>> groups = d.group_by("nonexistent_column")
        >>> len(groups)
        1  # returns a single copy of the original Data
        """
        if column not in self.data.columns:
            return [deepcopy(self)]

        s = self.data[column]

        if s.isna().all():
            return [deepcopy(self)]

        groups: List["Data"] = []

        for _, df_group in self.data.groupby(s, dropna=False):
            new_obj = deepcopy(self)
            new_obj.data = df_group.copy()
            groups.append(new_obj)

        return groups
