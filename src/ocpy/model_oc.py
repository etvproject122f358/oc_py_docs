from abc import abstractmethod, ABC
from logging import Logger

from typing import Dict, Self, Callable, List, Any
from lmfit.model import ModelResult

from pathlib import Path
import numpy as np
import pandas as pd

from .custom_types import ArrayReducer, BinarySeq


class ParameterModel(ABC):
    value: float
    min: float
    max: float
    std: float
    fixed: bool = False


class ModelComponentModel(ABC):
    @abstractmethod
    def __init__(self, args: List[ParameterModel] | None = None, logger: Logger | None = None) -> None:
        """Constructor method"""

    @abstractmethod
    def model_function(self) -> Any:
        """Definition of the function to fit"""


class OCModel(ABC):
    @classmethod
    @abstractmethod
    def from_file(cls, file: str | Path, columns: Dict[str, str] | None = None) -> Self:
        """Read data from file"""

    @abstractmethod
    def bin(
            self,
            bin_count: int = 1,
            bin_method: ArrayReducer | None = None,
            bin_error_method: ArrayReducer | None = None,
            bin_style: Callable[[pd.DataFrame, int], np.ndarray] | None = None
    ) -> Self:
        """Bins the data and returns each a new Self"""

    def __init__(
            self,
            minimum_time: List,
            minimum_time_error: List | None = None,
            weights: List | None = None,
            minimum_type: BinarySeq | None = None,
            labels: List | None = None,
            ecorr: List | None = None,
            oc: List | None = None, ):
        """Constructor method of oc class"""

    @abstractmethod
    def merge(self, oc: Self) -> Self:
        """Appends oc to this oc"""

    @abstractmethod
    def residue(self, coefficients: ModelResult) -> Self:
        """Removes the fit from current data"""

    @abstractmethod
    def fit(self, model_components: List[ModelComponentModel] | ModelComponentModel) -> ModelResult:
        """Fits the given ModelComponents to the O-C"""

    @abstractmethod
    def fit_keplerian(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a keplerian fit (also known as lite)"""

    @abstractmethod
    def fit_lite(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a lite fit (also known as keplerian fit)"""

    @abstractmethod
    def fit_linear(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a linear fit"""

    @abstractmethod
    def fit_quadratic(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a quadratic fit"""

    @abstractmethod
    def fit_sinusoidal(self, parameters: List[ParameterModel]) -> ModelComponentModel:
        """Makes a sinusoidal fit"""
