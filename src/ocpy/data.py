from logging import Logger
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Type, Any

from typing_extensions import Self, Literal

import pandas as pd
import numpy as np
from copy import deepcopy

from .model_data import DataModel
from .custom_types import BinarySeq
from .utils import Fixer
from .errors import LengthCheckError
from .oc import OC
from .oc_lmfit import OCLMFit
from .oc_pymc import OCPyMC


class Data(DataModel):
    """
    Container for eclipse minimum timing data.

    This class stores observed times of minima along with
    uncertainties, statistical weights, and classification
    flags. It provides tools for computing O−C residuals
    and filtering subsets of observations.

    The class is designed for use in period analysis,
    ETV studies, and dynamical modeling of eclipsing systems.
    """

    def __init__(
            self, minimum_time: List,
            minimum_time_error: List | None = None,
            weights: List | None = None,
            minimum_type: BinarySeq | None = None,
            labels: List | None = None,
            logger: Logger | None = None,
    ) -> None:
        """
        Initialize a Data object containing observed times of minima.

        This class represents a collection of observed eclipse minima times
        and their associated uncertainties, weights, and classifications.

        Parameters
        ----------
        minimum_time
            List of observed minimum times (e.g., in JD, BJD, or HJD).
        minimum_time_error
            Optional list of uncertainties associated with `minimum_time`.
            If provided, its length is fixed to match `minimum_time`.
        weights
            Optional list of statistical weights applied to each minimum.
        minimum_type
            Optional binary sequence indicating minimum type
            (e.g., primary = 0, secondary = 1).
        labels
            Optional list of labels associated with each minimum.
        logger
            Optional logger instance. If not provided, a module-level
            logger is created automatically.

        Notes
        -----
        This class does not enforce a specific time standard. It is the user's
        responsibility to ensure that all times are provided in a consistent
        reference frame.
        """

        fixed_minimum_time_error = Fixer.length_fixer(minimum_time_error, minimum_time)
        fixed_weights = Fixer.length_fixer(weights, minimum_time)
        fixed_minimum_type = Fixer.length_fixer(minimum_type, minimum_time)
        fixed_labels_to = Fixer.length_fixer(labels, minimum_time)

        self.data = pd.DataFrame(
            {
                "minimum_time": minimum_time,
                "minimum_time_error": fixed_minimum_time_error,
                "weights": fixed_weights,
                "minimum_type": fixed_minimum_type,
                "labels": fixed_labels_to,
            }
        )

        self.logger = Fixer.logger(logger, self.__class__.__name__)

    def __str__(self) -> str:
        """
        Return a string representation of the underlying DataFrame.
        """
        self.logger.debug("Getting string representation")

        return self.data.__str__()

    def __getitem__(self, item) -> Self | pd.Series:
        """
        Return a subset of the data.

        Parameters
        ----------
        item
            Either an integer index, a slice, or a boolean mask.

        Returns
        -------
        Data
            A new Data instance containing the selected rows.

        Notes
        -----
        This operation is non-mutating. The returned object is a new
        Data instance sharing no internal state with the original.
        The logger is propagated automatically.
        """
        self.logger.debug("Get item from the data")

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
                logger=self.logger
            )
        else:
            filtered_table = self.data[item]

            return Data(
                minimum_time=filtered_table["minimum_time"],
                minimum_time_error=filtered_table["minimum_time_error"],
                weights=filtered_table["weights"],
                minimum_type=filtered_table["minimum_type"],
                labels=filtered_table["labels"],
                logger=self.logger
            )

    def __setitem__(self, key, value) -> None:
        """
        Assign values to a column in the underlying DataFrame.

        Parameters
        ----------
        key
            Column name.
        value
            Values to assign.
        """
        self.logger.debug("Set item to the data")

        self.data.loc[:, key] = value

    def __len__(self) -> int:
        """
        Return the number of rows in the dataset.
        """
        self.logger.debug("Get length of the data")

        return len(self.data)

    @classmethod
    def from_file(cls, file: str | Path, columns: Dict[str, str] | None = None) -> Self:
        """
        Load minimum timing data from a file.

        Supported file formats are CSV and Excel (`.xls`, `.xlsx`).

        Parameters
        ----------
        file
            Path to the input file.
        columns
            Optional mapping for column renaming. Keys correspond to
            expected internal column names, values to file column names
            (or vice versa).

        Returns
        -------
        Data
            A new `Data` instance populated from the file.

        Raises
        ------
        ValueError
            If the file type is unsupported.
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

        kwargs = {c: (df[c] if c in df.columns else None) for c in expected}

        return cls(**kwargs)

    def _assign_or_fill(self, df: pd.DataFrame, col: str, values, override: bool = False) -> None:
        """
        Assign a column or fill it conditionally.

        Parameters
        ----------
        df
            Target pandas DataFrame.
        col
            Name of the column to assign or fill.
        values
            Value or sequence to assign.

        Notes
        -----
        If `value` is None, the column is filled with unity values.
        This behavior is intended for statistical weights or
        uncertainties when no explicit values are provided.
        """

        self.logger.info("Assign or conditionally fill a column on the data")

        if override or col not in df.columns:
            df[col] = values
        else:
            base = df[col]
            df[col] = base.where(~pd.isna(base), values)

    def fill_errors(self, errors: List | Tuple | np.ndarray | float, override: bool = False) -> Self:
        """
        Fill or replace minimum time errors.

        Parameters
        ----------
        errors
            Error values (scalar or sequence).
        override
            If True, overwrite existing values.

        Returns
        -------
        Data
            A new `Data` instance with updated errors.

        Raises
        ------
        LengthCheckError
            If the length of `errors` does not match the data length.
        """
        self.logger.info("Fill error columns on the data")

        new_data = deepcopy(self)
        if isinstance(errors, (list, tuple, np.ndarray)) and len(errors) != len(new_data.data):
            self.logger.error("Length of `errors` must be equal to the length of the data")
            raise LengthCheckError("Length of `errors` must be equal to the length of the data")

        self._assign_or_fill(new_data.data, "minimum_time_error", errors, override)
        return new_data

    def fill_weights(self, weights: List | Tuple | np.ndarray | float, override: bool = False) -> Self:
        """
        Fill or replace weights.

        Parameters
        ----------
        weights
            Weight values (scalar or sequence).
        override
            If True, overwrite existing values.

        Returns
        -------
        Data
            A new `Data` instance with updated weights.

        Raises
        ------
        LengthCheckError
            If the length of `weights` does not match the data length.
        """
        self.logger.info("Fill weights columns on the data")

        new_data = deepcopy(self)
        if isinstance(weights, (list, tuple, np.ndarray)) and len(weights) != len(new_data.data):
            self.logger.error("Length of `weights` must be equal to the length of the data")
            raise LengthCheckError("Length of `weights` must be equal to the length of the data")

        self._assign_or_fill(new_data.data, "weights", weights, override)
        return new_data

    def calculate_weights(self, method: Callable[[pd.Series], pd.Series] = None, override: bool = True) -> Self:
        """
        Calculate and assign weights from minimum time errors.

        By default, inverse-variance weighting is used:

        .. math::

            w = 1 / \\sigma^2

        Parameters
        ----------
        method
            Optional callable that takes a Series of errors and returns
            a Series of weights.
        override
            Whether to overwrite existing weights.

        Returns
        -------
        Data
            A new `Data` instance with computed weights.

        Raises
        ------
        ValueError
            If error values contain NaN or zero.
        TypeError
            If `method` is not callable.
        """
        self.logger.info("Trying to calculate weights")

        def inverse_variance_weights(err_days: pd.Series) -> np.ndarray | pd.Series:
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / np.square(err_days)

        new_data = deepcopy(self)
        minimum_time_error = new_data.data["minimum_time_error"]

        if minimum_time_error.hasnans:
            self.logger.error("minimum_time_error contains NaN value(s)")
            raise ValueError("minimum_time_error contains NaN value(s)")

        if (minimum_time_error == 0).any():
            self.logger.error("minimum_time_error contains `0`")
            raise ValueError("minimum_time_error contains `0`")

        if method is not None and not callable(method):
            self.logger.error("`method` must be callable or None for inverse variance weights")
            raise TypeError("`method` must be callable or None for inverse variance weights")

        if method is None:
            method = inverse_variance_weights

        weights = method(minimum_time_error)
        self._assign_or_fill(new_data.data, "weights", weights, override)
        return new_data

    def calculate_oc(self, reference_minimum: float, reference_period: float,
                     model_type: Literal["lmfit", "pymc", "oc"] = "lmfit") -> OC:
        """
        Compute Observed minus Calculated (O−C) residuals.

        This method computes eclipse timing residuals relative to a
        reference linear ephemeris and returns an O–C model object.

        Parameters
        ----------
        reference_minimum
            Reference epoch :math:`T_0`, corresponding to cycle number
            :math:`E = 0`.
        reference_period
            Orbital period :math:`P` of the system.
        model_type
            Output model type. One of ``"lmfit"``, ``"pymc"``, or ``"oc"``.

        Returns
        -------
        OC
            An O–C model instance containing cycle numbers and
            O–C residuals.

        Notes
        -----
        The O–C residual for each observed minimum is defined as:

        .. math::

            (O - C)_i = T_{\\mathrm{obs}, i} - (T_0 + E_i P)

        where:

        - :math:`T_{\\mathrm{obs}, i}` is the observed time of the *i*-th minimum
        - :math:`T_0` is the reference epoch
        - :math:`P` is the orbital period
        - :math:`E_i` is the integer (or half-integer) cycle number

        The cycle number is computed as:

        .. math::

            E_i = \\mathrm{round}\\left( \\frac{T_{\\mathrm{obs}, i} - T_0}{P} \\right)

        If a minimum is classified as secondary (via ``minimum_type``),
        the cycle number is shifted by half a cycle:

        .. math::

            E_i^{(\\mathrm{sec})} = \\mathrm{round}\\left( \\frac{T_{\\mathrm{obs}, i} - T_0}{P} - 0.5 \\right) + 0.5

        O–C diagrams are widely used to study period variations,
        apsidal motion, light-time effects, and dynamical perturbations
        in eclipsing binary systems.
        """
        self.logger.info("Calculating Observed - Calculated")

        df = self.data.copy()
        if "minimum_time" not in df.columns:
            self.logger.error("`minimum_time` column is required to compute O–C.")
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

        new_data: Dict[str, List[Any] | None] = {
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

        key = str(model_type).strip().lower()

        Target: Type[OC]

        if key in {"lmfit", "lmfit_model"}:
            Target = OCLMFit
        elif key in {"pymc", "pymc_model"}:
            Target = OCPyMC
        else:
            Target = OC

        return Target(**common_kwargs)

    def merge(self, data: Self) -> Self:
        """
        Merge this dataset with another `Data` instance.

        Parameters
        ----------
        data
            Another `Data` object.

        Returns
        -------
        Data
            A new `Data` instance containing concatenated rows.
        """
        self.logger.info("Merging two datasets")

        new_data = deepcopy(self)
        new_data.data = pd.concat([self.data, data.data], ignore_index=True, sort=False)
        return new_data

    def group_by(self, column: str | int) -> List["Data"]:
        """
        Split the dataset into groups based on a column.

        Parameters
        ----------
        column
            Column name to group by.

        Returns
        -------
        list of Data
            A list of `Data` objects, one per group.
        """
        self.logger.info("Grouping dataset")

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
