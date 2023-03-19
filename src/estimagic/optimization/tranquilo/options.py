from typing import NamedTuple

import numpy as np
from rich.table import Table
from rich.console import Console
from rich.style import Style


def get_default_radius_options(x):
    return RadiusOptions(initial_radius=0.1 * np.max(np.abs(x)))


def get_default_batch_size(n_cores):
    return n_cores


def get_default_acceptance_decider(noisy):
    return "noisy" if noisy else "classic"


def get_default_sample_size(model_type, x):
    if model_type == "quadratic":
        out = 2 * len(x) + 1
    else:
        out = len(x) + 1

    return out


def get_default_model_fitter(functype):
    return "tranquilo" if functype == "scalar" else "ols"


def get_default_subsolver(bounds, cube_subsolver, sphere_subsolver):
    return cube_subsolver if bounds.has_any else sphere_subsolver


def get_default_search_radius_factor(functype):
    return 4.25 if functype == "scalar" else 5.0


def get_default_model_type(functype):
    return "quadratic" if functype == "scalar" else "linear"


def get_default_aggregator(functype, model_type):
    if functype == "scalar" and model_type == "quadratic":
        aggregator = "identity"
    elif functype == "least_squares" and model_type == "linear":
        aggregator = "least_squares_linear"
    elif functype == "likelihood" and model_type == "linear":
        aggregator = "information_equality_linear"
    else:
        table = _get_aggregator_combination_table()
        Console().print(table)
        raise ValueError(
            "The requested combination of functype and model_type is not supported. "
            "Allowed combinations are listed in the table above. Default aggregators "
            "are colored green."
        )

    return aggregator


def _get_aggregator_combination_table():
    columns = ["function type", "model type", "aggregator"]
    rows = [
        ["scalar", "quadratic", "identity"],
        ["scalar", "quadratic", "sum"],
        ["least_squares", "linear", "least_squares_linear"],
        ["likelihood", "linear", "information_equality_linear"],
        ["likelihood", "linear", "sum"],
    ]
    table = Table(title="Allowed aggregator combinations")
    for col in columns:
        table.add_column(col)
    for k, row in enumerate(rows):
        end_section = True if k in (1, 2) else False
        style = Style(bold=True, color="bright_green") if k in (0, 2, 3) else None
        table.add_row(*row, style=style, end_section=end_section)
    return table


def get_default_n_evals_at_start(noisy):
    return 5 if noisy else 1


class StopOptions(NamedTuple):
    """Criteria for stopping without successful convergence."""

    max_iter: int
    max_eval: int
    max_time: float


class ConvOptions(NamedTuple):
    """Criteria for successful convergence."""

    disable: bool
    ftol_abs: float
    gtol_abs: float
    xtol_abs: float
    ftol_rel: float
    gtol_rel: float
    xtol_rel: float
    min_radius: float


class RadiusOptions(NamedTuple):
    """Options for trust-region radius management."""

    initial_radius: float
    min_radius: float = 1e-6
    max_radius: float = 1e6
    rho_decrease: float = 0.1
    rho_increase: float = 0.1
    shrinking_factor: float = 0.5
    expansion_factor: float = 2.0
    large_step: float = 0.5
    max_radius_to_step_ratio: float = np.inf


class AcceptanceOptions(NamedTuple):
    confidence_level: float = 0.8
    power_level: float = 0.8
    n_initial: int = 5
    n_min: int = 5
    n_max: int = 100
    min_improvement: float = 0.0


class StagnationOptions(NamedTuple):
    min_relative_step_keep: float = 0.125
    min_relative_step: float = 0.05
    sample_increment: int = 1
    max_trials: int = 1
    drop: bool = True


class SubsolverOptions(NamedTuple):
    maxiter: int = 20
    maxiter_gradient_descent: int = 5
    conjugate_gradient_method: str = "cg"
    gtol_abs: float = 1e-8
    gtol_rel: float = 1e-8
    gtol_scaled: float = 0.0
    gtol_abs_conjugate_gradient: float = 1e-8
    gtol_rel_conjugate_gradient: float = 1e-6
    k_easy: float = 0.1
    k_hard: float = 0.2


class FitterOptions(NamedTuple):
    model_type: str
    l2_penalty_linear: float = 0.0
    l2_penalty_square: float = 0.1
    p_intercept: float = 0.05
    p_linear: float = 0.4
    p_square: float = 1.0


class VarianceEstimatorOptions(NamedTuple):
    max_distance_factor: float = 3.0
    min_n_evals: int = 3
