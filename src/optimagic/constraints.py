from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Any, Callable

import pandas as pd
from numpy.typing import ArrayLike

from optimagic.exceptions import InvalidConstraintError
from optimagic.typing import PyTree


class Constraint(ABC):
    """Base class for all constraints used for subtyping."""

    selector: Callable[[PyTree], PyTree]

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass


def identity_selector(x: PyTree) -> PyTree:
    return x


@dataclass(frozen=True)
class FixedConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "fixed", "selector": self.selector}


@dataclass(frozen=True)
class IncreasingConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "increasing", "selector": self.selector}


@dataclass(frozen=True)
class DecreasingConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "decreasing", "selector": self.selector}


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "equality", "selector": self.selector}


@dataclass(frozen=True)
class PairwiseEqualityConstraint(Constraint):
    selectors: list[Callable[[PyTree], PyTree]]

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "pairwise_equality", "selectors": self.selectors}


@dataclass(frozen=True)
class ProbabilityConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "probability", "selector": self.selector}


@dataclass(frozen=True)
class FlatCovarianceConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    bounds_distance: float | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "covariance",
            "selector": self.selector,
            **_select_non_none(bounds_distance=self.bounds_distance),
        }


@dataclass(frozen=True)
class FlatSDCorrConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    bounds_distance: float | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "sdcorr",
            "selector": self.selector,
            **_select_non_none(bounds_distance=self.bounds_distance),
        }


@dataclass(frozen=True)
class LinearConstraint(Constraint):
    selector: Callable[[PyTree], ArrayLike | "pd.Series[float]" | float] = (
        identity_selector
    )
    _: KW_ONLY
    weights: ArrayLike | "pd.Series[float]" | float | pd.DataFrame | None = None
    lower_bound: ArrayLike | "pd.Series[float]" | float | None = None
    upper_bound: ArrayLike | "pd.Series[float]" | float | None = None
    value: ArrayLike | "pd.Series[float]" | float | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "linear",
            "selector": self.selector,
            "weights": self.weights,
            **_select_non_none(
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                value=self.value,
            ),
        }

    def __post_init__(self) -> None:
        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )


@dataclass(frozen=True)
class NonlinearConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    func: Callable[[PyTree], ArrayLike | "pd.Series[float]" | float] | None = None
    derivative: Callable[[PyTree], PyTree] | None = None
    lower_bound: ArrayLike | "pd.Series[float]" | float | None = None
    upper_bound: ArrayLike | "pd.Series[float]" | float | None = None
    value: ArrayLike | "pd.Series[float]" | float | None = None
    tol: float | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "nonlinear",
            "selector": self.selector,
            **_select_non_none(
                func=self.func,
                derivative=self.derivative,
                # In the dict representation, we write _bounds instead of _bound.
                lower_bounds=self.lower_bound,
                upper_bounds=self.upper_bound,
                value=self.value,
                tol=self.tol,
            ),
        }

    def __post_init__(self) -> None:
        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )

        if self.tol is not None and self.tol < 0:
            raise InvalidConstraintError("'tol' must be non-negative.")

        if self.func is None or not callable(self.func):
            raise InvalidConstraintError("'func' must be callable.")


def pre_process_constraints(
    constraints: list[Constraint | dict[str, Any]] | Constraint | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if constraints is None:
        out = []
    elif isinstance(constraints, dict):
        out = [constraints]
    elif isinstance(constraints, Constraint):
        out = [constraints._to_dict()]
    elif isinstance(constraints, list):
        out = [c._to_dict() if isinstance(c, Constraint) else c for c in constraints]
    else:
        msg = (
            f"Invalid constraints type: {type(constraints)}. Must be a constraint "
            "object or list thereof imported from `optimagic.constraints`."
        )
        raise InvalidConstraintError(msg)

    return out


def _all_none(*args: Any) -> bool:
    return all(v is None for v in args)


def _select_non_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
