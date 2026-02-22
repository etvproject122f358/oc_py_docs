from typing import Union, List, Tuple, Literal, Callable, Any

import numpy as np
from numpy._typing import NDArray

# A container (List, Tuple, ndarray) that contains only 1s and 0s
BinarySeq = Union[
    List[Literal[0, 1]],
    Tuple[Literal[0, 1], ...],
    NDArray[np.bool_],
]
"""Type alias for a binary sequence (0/1 or bool), typically used for minimum classifications."""

# A callable that takes a 1D ndarray and returns a numeric value
ArrayReducer = Callable[[NDArray[np.floating]], int | float | np.number]
"""Type alias for a function that reduces an array to a single number (e.g., mean, median)."""

NumberOrParam = float | int | None | Any
"""Type alias for a value that can be either a numeric literal or a Parameter instance."""
