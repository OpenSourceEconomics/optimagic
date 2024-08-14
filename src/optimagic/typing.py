from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, ItemsView, Iterator, KeysView, Literal, ValuesView

import numpy as np
from numpy._typing import NDArray

PyTree = Any
PyTreeRegistry = dict[type | str, dict[str, Callable[[Any], Any]]]
Scalar = Any


class AggregationLevel(Enum):
    """Enum to specify the aggregation level of objective functions and solvers."""

    SCALAR = "scalar"
    LEAST_SQUARES = "least_squares"
    LIKELIHOOD = "likelihood"


@dataclass(frozen=True)
class DictLikeAccess:
    """Useful base class for replacing string-based dictionaries with dataclass
    instances and keeping backward compatability regarding read access to the data
    structure."""

    def __getitem__(self, key: str) -> Any:
        if key in self.__dict__:
            return getattr(self, key)
        else:
            raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict_repr())

    def _dict_repr(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def keys(self) -> KeysView[str]:
        return self._dict_repr().keys()

    def items(self) -> ItemsView[str, Any]:
        return self._dict_repr().items()

    def values(self) -> ValuesView[str]:
        return self._dict_repr().values()


@dataclass(frozen=True)
class TupleLikeAccess:
    """Useful base class for replacing tuples with dataclass instances and keeping
    backward compatability regarding read access to the data structure."""

    def __getitem__(self, index: int | slice) -> Any:
        field_values = [getattr(self, field.name) for field in fields(self)]
        return field_values[index]

    def __len__(self) -> int:
        return len(fields(self))

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            yield getattr(self, field.name)


class OptimizationType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


OptimizationTypeLiteral = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class IterationHistory(DictLikeAccess):
    """History of iterations in a process.

    Attributes:
        params: A list of parameters used in each iteration.
        criterion: A list of criterion values obtained in each iteration.
        runtime: A list or array of runtimes associated with each iteration.

    """

    params: list[PyTree]
    criterion: list[float]
    runtime: list[float] | NDArray[np.float64]


@dataclass(frozen=True)
class MultiStartIterationHistory(TupleLikeAccess):
    """History of multiple start iterations.

    Attributes:
        history: The main iteration history, representing the best end value.
        local_histories: Optional, a list of local iteration histories.
        exploration: Optional, iteration history for exploration steps.

    """

    history: IterationHistory
    local_histories: list[IterationHistory] | None = None
    exploration: IterationHistory | None = None
