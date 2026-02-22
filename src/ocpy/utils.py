from logging import Logger, getLogger

import numpy as np

from .errors import LengthCheckError


class Checker:
    @staticmethod
    def length_checker(data, reference):
        """
        Check if two sequences have the same length.

        Parameters
        ----------
        data : Sized
            The sequence to check.
        reference : Sized
            The sequence to compare against.

        Raises
        ------
        LengthCheckError
            If lengths do not match.
        """
        if len(reference) != len(data):
            raise LengthCheckError("length of data is not sufficient")


class Fixer:
    @staticmethod
    def length_fixer(data, reference):
        """
        Ensure data matches the length of a reference sequence.

        If data is a scalar or a string, it is broadcast to the
        reference length. If it is already a sequence, its length
        is verified.

        Parameters
        ----------
        data : Any
            The data to fix or verify.
        reference : Sized | None
            The reference sequence.

        Returns
        -------
        np.ndarray | Any
            Fixed or broadcasted data.
        """
        if reference is None:
            return data

        if isinstance(data, str):
            return np.array([data] * len(reference), dtype=object)

        if hasattr(data, "__len__"):
            Checker.length_checker(data, reference)
            if isinstance(data, list):
                return np.array(data)
            return data
        else:
            return np.array([data] * len(reference))

    @staticmethod
    def none_to_nan(data_frame):
        """
        Replace None values with np.nan in a DataFrame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            The target DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with None replaced by NaN.
        """
        return data_frame.replace({None: np.nan})

    @staticmethod
    def logger(logger: Logger | None = None, name: str | None = None) -> Logger:
        """
        Checks if a logger is passed as an argument. If not, it returns a logger with the specified name
        or a default name.


        Parameters
        ----------
        logger: Logger, default = None
            An optional Logger instance.
        name: str, default = None
            An optional string representing the name of the logger.

        Returns
        -------
        Logger
            The initialized or provided Logger instance.
        """
        if logger is None:
            if name is None:
                return getLogger("OCPY")
            else:
                return getLogger(name)
        else:
            return logger
