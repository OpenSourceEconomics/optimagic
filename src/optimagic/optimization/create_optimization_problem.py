import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

from optimagic import deprecations
from optimagic.algorithms import ALL_ALGORITHMS
from optimagic.deprecations import (
    handle_log_options_throw_deprecated_warning,
    replace_and_warn_about_deprecated_algo_options,
    replace_and_warn_about_deprecated_bounds,
)
from optimagic.differentiation.numdiff_options import (
    NumdiffOptions,
    NumdiffPurpose,
    get_default_numdiff_options,
    pre_process_numdiff_options,
)
from optimagic.exceptions import (
    AliasError,
    InvalidFunctionError,
    MissingInputError,
)
from optimagic.logging.logger import LogOptions, SQLiteLogOptions
from optimagic.optimization.algorithm import AlgoInfo, Algorithm
from optimagic.optimization.fun_value import (
    SpecificFunctionValue,
    convert_fun_output_to_function_value,
    enforce_return_type,
    enforce_return_type_with_jac,
)
from optimagic.optimization.multistart_options import (
    MultistartOptions,
    pre_process_multistart,
)
from optimagic.optimization.scipy_aliases import (
    map_method_to_algorithm,
    split_fun_and_jac,
)
from optimagic.parameters.bounds import Bounds, pre_process_bounds
from optimagic.parameters.scaling import ScalingOptions, pre_process_scaling
from optimagic.shared.process_user_function import (
    get_kwargs_from_args,
    infer_aggregation_level,
    partial_func_of_params,
)
from optimagic.typing import AggregationLevel, Direction, ErrorHandling, PyTree
from optimagic.utilities import propose_alternatives


@dataclass(frozen=True)
class OptimizationProblem:
    """Collect everything that defines the optimization problem.

    The attributes are very close to the arguments of `maximize` and `minimize` but they
    are converted to stricter types. For example, the bounds argument that can be a
    sequence of tuples, a scipy.optimize.Bounds object or an optimagic.Bounds when
    calling `maximize` or `minimize` is converted to an optimagic.Bounds object.

    All deprecated arguments are removed and all scipy aliases are replaced by their
    optimagic counterparts.

    All user provided functions are partialled if corresponding `kwargs` dictionaries
    were provided.

    # TODO: Document attributes after other todos are resolved.

    """

    fun: Callable[[PyTree], SpecificFunctionValue]
    params: PyTree
    algorithm: Algorithm
    bounds: Bounds | None
    # TODO: Only allow list[Constraint] or Constraint
    constraints: list[dict[str, Any]]
    jac: Callable[[PyTree], PyTree] | None
    fun_and_jac: Callable[[PyTree], tuple[SpecificFunctionValue, PyTree]] | None
    numdiff_options: NumdiffOptions
    # TODO: logging will become None | Logger and log_options will be removed
    error_handling: ErrorHandling
    logging: LogOptions | None
    error_penalty: dict[str, Any] | None
    scaling: ScalingOptions | None
    multistart: MultistartOptions | None
    collect_history: bool
    skip_checks: bool
    direction: Direction
    fun_eval: SpecificFunctionValue


def create_optimization_problem(
    direction,
    fun,
    params,
    algorithm,
    *,
    bounds,
    fun_kwargs,
    constraints,
    algo_options,
    jac,
    jac_kwargs,
    fun_and_jac,
    fun_and_jac_kwargs,
    numdiff_options,
    logging,
    error_handling,
    error_penalty,
    scaling,
    multistart,
    collect_history,
    skip_checks,
    # scipy aliases
    x0,
    method,
    args,
    # scipy arguments that are not yet supported
    hess,
    hessp,
    callback,
    # scipy arguments that will never be supported
    options,
    tol,
    # deprecated arguments
    criterion,
    criterion_kwargs,
    derivative,
    derivative_kwargs,
    criterion_and_derivative,
    criterion_and_derivative_kwargs,
    lower_bounds,
    log_options,
    upper_bounds,
    soft_lower_bounds,
    soft_upper_bounds,
    scaling_options,
    multistart_options,
):
    # ==================================================================================
    # error handling needed as long as fun is an optional argument
    # ==================================================================================

    if fun is None and criterion is None:
        msg = (
            "Missing objective function. Please provide an objective function as the "
            "first positional argument or as the keyword argument `fun`."
        )
        raise MissingInputError(msg)

    if params is None and x0 is None:
        msg = (
            "Missing start parameters. Please provide start parameters as the second "
            "positional argument or as the keyword argument `params`."
        )
        raise MissingInputError(msg)

    if algorithm is None and method is None:
        msg = (
            "Missing algorithm. Please provide an algorithm as the third positional "
            "argument or as the keyword argument `algorithm`."
        )
        raise MissingInputError(msg)

    # ==================================================================================
    # deprecations
    # ==================================================================================

    if log_options is not None:
        logging = handle_log_options_throw_deprecated_warning(log_options, logging)

    if criterion is not None:
        deprecations.throw_criterion_future_warning()
        fun = criterion if fun is None else fun

    if criterion_kwargs is not None:
        deprecations.throw_criterion_kwargs_future_warning()
        fun_kwargs = criterion_kwargs if fun_kwargs is None else fun_kwargs

    if derivative is not None:
        deprecations.throw_derivative_future_warning()
        jac = derivative if jac is None else jac

    if derivative_kwargs is not None:
        deprecations.throw_derivative_kwargs_future_warning()
        jac_kwargs = derivative_kwargs if jac_kwargs is None else jac_kwargs

    if criterion_and_derivative is not None:
        deprecations.throw_criterion_and_derivative_future_warning()
        fun_and_jac = criterion_and_derivative if fun_and_jac is None else fun_and_jac

    if criterion_and_derivative_kwargs is not None:
        deprecations.throw_criterion_and_derivative_kwargs_future_warning()
        fun_and_jac_kwargs = (
            criterion_and_derivative_kwargs
            if fun_and_jac_kwargs is None
            else fun_and_jac_kwargs
        )

    if scaling_options is not None:
        deprecations.throw_scaling_options_future_warning()
        if scaling is True and scaling_options is not None:
            scaling = scaling_options

    if multistart_options is not None:
        deprecations.throw_multistart_options_future_warning()
        if multistart is True and multistart_options is not None:
            multistart = multistart_options

    deprecations.throw_dict_constraints_future_warning_if_required(constraints)

    algo_options = replace_and_warn_about_deprecated_algo_options(algo_options)

    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
    )

    if isinstance(jac, dict):
        jac = deprecations.replace_and_warn_about_deprecated_derivatives(jac, "jac")

    if isinstance(fun_and_jac, dict):
        fun_and_jac = deprecations.replace_and_warn_about_deprecated_derivatives(
            fun_and_jac, "fun_and_jac"
        )
    # ==================================================================================
    # handle scipy aliases
    # ==================================================================================

    if x0 is not None:
        if params is not None:
            msg = (
                "x0 is an alias for params (for better compatibility with scipy). "
                "Do not use both x0 and params."
            )
            raise AliasError(msg)
        else:
            params = x0

    if method is not None:
        if algorithm is not None:
            msg = (
                "method is an alias for algorithm to select the scipy optimizers under "
                "their original name. Do not use both method and algorithm."
            )
            raise AliasError(msg)
        else:
            algorithm = map_method_to_algorithm(method)

    if args is not None:
        if (
            fun_kwargs is not None
            or jac_kwargs is not None
            or fun_and_jac_kwargs is not None
        ):
            msg = (
                "args is an alternative to fun_kwargs, jac_kwargs and "
                "fun_and_jac_kwargs that optimagic supports for compatibility "
                "with scipy. Do not use args in conjunction with any of the other "
                "arguments."
            )
            raise AliasError(msg)
        else:
            kwargs = get_kwargs_from_args(args, fun, offset=1)
            fun_kwargs, jac_kwargs, fun_and_jac_kwargs = kwargs, kwargs, kwargs

    # jac is not an alias but we need to handle the case where `jac=True`, i.e. fun is
    # actually fun_and_jac. This is not recommended in optimagic because then optimizers
    # cannot evaluate fun in isolation but we can easily support it for compatibility.
    if jac is True:
        jac = None
        if fun_and_jac is None:
            fun_and_jac = fun
            fun = split_fun_and_jac(fun_and_jac, target="fun")

    # ==================================================================================
    # Handle scipy arguments that are not yet implemented
    # ==================================================================================

    if hess is not None:
        msg = (
            "The hess argument is not yet supported in optimagic. Creat an issue on "
            "https://github.com/optimagic-dev/optimagic/ if you have urgent need "
            "for this feature."
        )
        raise NotImplementedError(msg)

    if hessp is not None:
        msg = (
            "The hessp argument is not yet supported in optimagic. Creat an issue on "
            "https://github.com/optimagic-dev/optimagic/ if you have urgent need "
            "for this feature."
        )
        raise NotImplementedError(msg)

    if callback is not None:
        msg = (
            "The callback argument is not yet supported in optimagic. Creat an issue "
            "on https://github.com/optimagic-dev/optimagic/ if you have urgent "
            "need for this feature."
        )
        raise NotImplementedError(msg)

    # ==================================================================================
    # Handle scipy arguments that will never be supported
    # ==================================================================================

    if options is not None:
        # TODO: Add link to a how-to guide or tutorial for this
        msg = (
            "The options argument is not supported in optimagic. Please use the "
            "algo_options argument instead."
        )
        raise NotImplementedError(msg)

    if tol is not None:
        # TODO: Add link to a how-to guide or tutorial for this
        msg = (
            "The tol argument is not supported in optimagic. Please use "
            "algo_options or configured algorithms instead to set convergence criteria "
            "for your optimizer."
        )
        raise NotImplementedError(msg)

    # ==================================================================================
    # Convert literals to enums
    # ==================================================================================
    error_handling = ErrorHandling(error_handling)

    # ==================================================================================
    # Set default values and check options
    # ==================================================================================
    bounds = pre_process_bounds(bounds)
    scaling = pre_process_scaling(scaling)
    multistart = pre_process_multistart(multistart)
    numdiff_options = pre_process_numdiff_options(numdiff_options)
    constraints = deprecations.pre_process_constraints(constraints)

    if numdiff_options is None:
        numdiff_options = get_default_numdiff_options(purpose=NumdiffPurpose.OPTIMIZE)

    fun_kwargs = {} if fun_kwargs is None else fun_kwargs
    constraints = [] if constraints is None else constraints
    algo_options = {} if algo_options is None else algo_options
    jac_kwargs = {} if jac_kwargs is None else jac_kwargs
    fun_and_jac_kwargs = {} if fun_and_jac_kwargs is None else fun_and_jac_kwargs
    error_penalty = {} if error_penalty is None else error_penalty

    if isinstance(logging, str) or isinstance(logging, Path):
        log_path = Path(logging)
        logging = SQLiteLogOptions(log_path)

    # ==================================================================================
    # evaluate fun for the first time
    # ==================================================================================
    fun = partial_func_of_params(
        func=fun,
        kwargs=fun_kwargs,
        name="criterion",
        skip_checks=skip_checks,
    )

    # This should be done as late as possible; It has to be done here to infer the
    # problem type until the decorator approach becomes mandatory.
    # TODO: Move this into `_optimize` as soon as we reach 0.6.0
    try:
        fun_eval = fun(params)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating fun at start params."
        raise InvalidFunctionError(msg) from e

    if deprecations.is_dict_output(fun_eval):
        deprecations.throw_dict_output_warning()

    # ==================================================================================
    # infer the problem type
    # ==================================================================================

    if deprecations.is_dict_output(fun_eval):
        problem_type = deprecations.infer_problem_type_from_dict_output(fun_eval)
    else:
        problem_type = infer_aggregation_level(fun)

    if (
        problem_type == AggregationLevel.LEAST_SQUARES
        and direction == Direction.MAXIMIZE
    ):
        raise InvalidFunctionError("Least-squares problems cannot be maximized.")

    # ==================================================================================
    # process the fun_eval; Can be removed once the first evaluation gets moved to
    # a later point where the `enforce` decorator has already been applied.
    # ==================================================================================
    if deprecations.is_dict_output(fun_eval):
        fun_eval = deprecations.convert_dict_to_function_value(fun_eval)
        fun = deprecations.replace_dict_output(fun)
    else:
        fun_eval = convert_fun_output_to_function_value(fun_eval, problem_type)

    fun = enforce_return_type(problem_type)(fun)

    # ==================================================================================
    # Process the user provided algorithm
    # ==================================================================================

    algorithm = pre_process_user_algorithm(algorithm)
    algorithm = algorithm.with_option_if_applicable(**algo_options)

    if algorithm.algo_info.solver_type == AggregationLevel.LIKELIHOOD:
        if problem_type not in [
            AggregationLevel.LIKELIHOOD,
            AggregationLevel.LEAST_SQUARES,
        ]:
            raise InvalidFunctionError(
                "Likelihood solvers can only be used with likelihood or least-squares "
                "problems."
            )
    elif algorithm.algo_info.solver_type == AggregationLevel.LEAST_SQUARES:
        if problem_type != AggregationLevel.LEAST_SQUARES:
            raise InvalidFunctionError(
                "Least-squares solvers can only be used with least-squares problems."
            )

    # ==================================================================================
    # select the correct derivative functions
    # ==================================================================================

    if jac is not None:
        jac = pre_process_derivatives(
            candidate=jac, name="jac", solver_type=algorithm.algo_info.solver_type
        )

    if fun_and_jac is not None:
        fun_and_jac = pre_process_derivatives(
            candidate=fun_and_jac,
            name="fun_and_jac",
            solver_type=algorithm.algo_info.solver_type,
        )

    # ==================================================================================
    # partial the kwargs into corresponding functions
    # ==================================================================================

    if jac is not None:
        jac = partial_func_of_params(
            func=jac,
            kwargs=jac_kwargs,
            name="derivative",
            skip_checks=skip_checks,
        )

    if fun_and_jac is not None:
        fun_and_jac = partial_func_of_params(
            func=fun_and_jac,
            kwargs=fun_and_jac_kwargs,
            name="criterion_and_derivative",
            skip_checks=skip_checks,
        )
        fun_and_jac = deprecations.replace_dict_output(fun_and_jac)

        fun_and_jac = enforce_return_type_with_jac(algorithm.algo_info.solver_type)(
            fun_and_jac
        )

    # ==================================================================================
    # Check types of arguments
    # ==================================================================================

    if not skip_checks:
        if params is None:
            raise ValueError("params cannot be None")

        if not isinstance(fun, Callable):
            raise ValueError("fun must be a callable")

        if not isinstance(algorithm, Algorithm):
            raise ValueError("algorithm must be an Algorithm object.")

        if not isinstance(algo_options, dict | None):
            raise ValueError("algo_options must be a dictionary or None")

        if not isinstance(algorithm.algo_info, AlgoInfo):
            raise ValueError("algo_info must be an AlgoInfo object")

        if not isinstance(bounds, Bounds | None):
            raise ValueError("bounds must be a Bounds object or None")

        if not all(isinstance(c, dict) for c in constraints):
            # TODO: Only allow list[Constraint]
            raise ValueError("constraints must be a list of dictionaries")

        if not isinstance(jac, Callable | None):
            raise ValueError("jac must be a callable or None")

        if not isinstance(fun_and_jac, Callable | None):
            raise ValueError("fun_and_jac must be a callable or None")

        if not isinstance(numdiff_options, NumdiffOptions):
            raise ValueError("numdiff_options must be a NumdiffOptions object")

        if not isinstance(logging, bool | Path | LogOptions | None):
            raise ValueError(
                "logging must be a boolean, a path, a LogOptions instance or None"
            )

        if not isinstance(log_options, dict | None):
            raise ValueError("log_options must be a dictionary or None")

        if not isinstance(error_penalty, dict | None):
            raise ValueError("error_penalty must be a dictionary or None")

        if not isinstance(scaling, ScalingOptions | None):
            raise ValueError("scaling must be a ScalingOptions object or None")

        if not isinstance(multistart, MultistartOptions | None):
            raise ValueError("multistart must be a MultistartOptions object or None")

        if not isinstance(collect_history, bool):
            raise ValueError("collect_history must be a boolean")

    # ==================================================================================
    # create the problem object
    # ==================================================================================

    problem = OptimizationProblem(
        fun=fun,
        params=params,
        algorithm=algorithm,
        bounds=bounds,
        constraints=constraints,
        jac=jac,
        fun_and_jac=fun_and_jac,
        numdiff_options=numdiff_options,
        logging=logging,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        direction=direction,
        fun_eval=fun_eval,
    )

    return problem


def pre_process_derivatives(candidate, name, solver_type):
    if callable(candidate):
        candidate = [candidate]

    out = None
    for func in candidate:
        if not callable(func):
            raise ValueError(f"{name} must be a callable or sequence of callables.")

        problem_type = infer_aggregation_level(func)
        if problem_type == solver_type:
            out = func

    if out is None:
        msg = (
            f"You used the `{name}` argument but none of the callables you provided "
            "has the correct aggregation level for your selected optimization "
            "algorithm. Falling back to numerical derivatives."
        )
        warnings.warn(msg)

    return out


def pre_process_user_algorithm(
    algorithm: str | Algorithm | Type[Algorithm],
) -> Algorithm:
    """Process the user specfied algorithm."""
    if isinstance(algorithm, str):
        try:
            # Use ALL_ALGORITHMS and not just AVAILABLE_ALGORITHMS such that the
            # algorithm specific error message with installation instruction will be
            # reached if an optional dependency is not installed.
            algorithm = ALL_ALGORITHMS[algorithm]()
        except KeyError:
            proposed = propose_alternatives(algorithm, list(ALL_ALGORITHMS))
            raise ValueError(
                f"Invalid algorithm: {algorithm}. Did you mean {proposed}?"
            ) from None
    elif isinstance(algorithm, type) and issubclass(algorithm, Algorithm):
        algorithm = algorithm()

    return algorithm
