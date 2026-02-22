from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from typing_extensions import Self

import pandas as pd
from numpy._typing import NDArray

from .custom_types import BinarySeq


class DataModel(ABC):
    @abstractmethod
    def __init__(
            self,
            minimum_time: List,
            minimum_time_error: List | None = None,
            weights: List | None = None,
            minimum_type: BinarySeq | None = None,
            labels: List | None = None,
            ecorr: List | None = None,
            oc: List | None = None,
    ) -> None:
        """Constructor method of data class"""

    @abstractmethod
    def __getitem__(self, item) -> Self | pd.Series:
        """Get item works"""

    @classmethod
    @abstractmethod
    def from_file(cls, file: str | Path, columns: Dict[str, str] | None = None) -> Self:
        """Read data from file"""

    @abstractmethod
    def _assign_or_fill(self, df: pd.DataFrame, col: str, values, override: bool) -> None:
        """Assign new values to given col's"""

    @abstractmethod
    def fill_errors(self, errors: List | Tuple | NDArray | float, override: bool = False) -> Self:
        """Fills th errors"""

    @abstractmethod
    def fill_weights(self, weights: List | Tuple | NDArray | float, override: bool = False) -> Self:
        """Fills th weights"""

    @abstractmethod
    def calculate_weights(
            self,
            method: Callable[[pd.Series], pd.Series] | None = None,
            override: bool = True
    ) -> Self:
        """Calculates weights using errors"""

    @abstractmethod
    def calculate_oc(self, reference_period: float, reference_minimum: float) -> Self:
        """Calculates the O-C for this Data"""

    @abstractmethod
    def merge(self, data: Self) -> Self:
        """Appends data to this DataModel"""

    @abstractmethod
    def group_by(self, column: str | int) -> List[Self]:
        """Group data by column's data"""
