"""Public functions for optimization.

This module defines the public functions `maximize` and `minimize` that will be called
by users.

Internally, `maximize` and `minimize` just call `create_optimization_problem` with
all arguments and add the `direction`. In `create_optimization_problem`, the user input
is consolidated and converted to stricter types.  The resulting `OptimizationProblem`
is then passed to `_optimize` which handles the optimization logic.

`_optimize` processes the optimization problem and performs the actual optimization.

"""

from typing import Any, cast

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.exceptions import (
    InvalidFunctionError,
)
from optimagic.logging.logger import LogReader, LogStore
from optimagic.logging.types import ProblemInitialization
from optimagic.optimization.create_optimization_problem import (
    OptimizationProblem,
    create_optimization_problem,
)
from optimagic.optimization.error_penalty import get_error_penalty_function
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.optimization.multistart import (
    run_multistart_optimization,
)
from optimagic.optimization.multistart_options import (
    get_internal_multistart_options_from_public,
)
from optimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.optimization.process_results import (
    ExtraResultFields,
    process_multistart_result,
    process_single_result,
)
from optimagic.parameters.conversion import (
    get_converter,
)
from optimagic.parameters.nonlinear_constraints import process_nonlinear_constraints
from optimagic.typing import AggregationLevel, Direction


def maximize(
    fun=None,
    params=None,
    algorithm=None,
    *,
    bounds=None,
    constraints=None,
    fun_kwargs=None,
    algo_options=None,
    jac=None,
    jac_kwargs=None,
    fun_and_jac=None,
    fun_and_jac_kwargs=None,
    numdiff_options=None,
    logging=None,
    error_handling="raise",
    error_penalty=None,
    scaling=False,
    multistart=False,
    collect_history=True,
    skip_checks=False,
    # scipy aliases
    x0=None,
    method=None,
    args=None,
    # scipy arguments that are not yet supported
    hess=None,
    hessp=None,
    callback=None,
    # scipy arguments that will never be supported
    options=None,
    tol=None,
    # deprecated arguments
    criterion=None,
    criterion_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
    log_options=None,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    scaling_options=None,
    multistart_options=None,
):
    """Maximize fun using algorithm subject to constraints.

    TODO: Write docstring after enhancement proposals are implemented.

    Args:
        fun: The objective function of a scalar, least-squares or likelihood
            optimization problem. Non-scalar objective functions have to be marked
            with the `mark.likelihood` or `mark.least_squares` decorators. `fun` maps
            params and fun_kwargs to an objective value.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.

    """
    problem = create_optimization_problem(
        direction=Direction.MAXIMIZE,
        fun=fun,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        scaling_options=scaling_options,
        multistart_options=multistart_options,
    )
    return _optimize(problem)


def minimize(
    fun=None,
    params=None,
    algorithm=None,
    *,
    bounds=None,
    constraints=None,
    fun_kwargs=None,
    algo_options=None,
    jac=None,
    jac_kwargs=None,
    fun_and_jac=None,
    fun_and_jac_kwargs=None,
    numdiff_options=None,
    logging=None,
    error_handling="raise",
    error_penalty=None,
    scaling=False,
    multistart=False,
    collect_history=True,
    skip_checks=False,
    # scipy aliases
    x0=None,
    method=None,
    args=None,
    # scipy arguments that are not yet supported
    hess=None,
    hessp=None,
    callback=None,
    # scipy arguments that will never be supported
    options=None,
    tol=None,
    # deprecated arguments
    criterion=None,
    criterion_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
    lower_bounds=None,
    log_options=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    scaling_options=None,
    multistart_options=None,
):
    """Minimize criterion using algorithm subject to constraints.

    TODO: Write docstring after enhancement proposals are implemented.

    Args:
        fun: The objective function of a scalar or likelihood optimization problem.
            Non-scalar objective functions have to be marked with the `mark.likelihood`
            decorator. `fun` maps params and fun_kwargs to an objective value.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.

    """

    problem = create_optimization_problem(
        direction=Direction.MINIMIZE,
        fun=fun,
        params=params,
        algorithm=algorithm,
        bounds=bounds,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        log_options=log_options,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        scaling_options=scaling_options,
        multistart_options=multistart_options,
    )
    return _optimize(problem)


def _optimize(problem: OptimizationProblem) -> OptimizeResult:
    """Solve an optimization problem."""
    # ==================================================================================
    # Split constraints into nonlinear and reparametrization parts
    # ==================================================================================
    constraints = problem.constraints

    nonlinear_constraints = [c for c in constraints if c["type"] == "nonlinear"]

    if nonlinear_constraints:
        if not problem.algorithm.algo_info.supports_nonlinear_constraints:
            raise ValueError(
                f"Algorithm {problem.algorithm.name} does not support "
                "nonlinear constraints."
            )

    # the following constraints will be handled via reparametrization
    constraints = [c for c in constraints if c["type"] != "nonlinear"]

    # ==================================================================================
    # Do first evaluation of user provided functions
    # ==================================================================================
    first_crit_eval = problem.fun_eval

    # do first derivative evaluation (if given)
    if problem.jac is not None:
        try:
            first_deriv_eval = problem.jac(problem.params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at start params."
            raise InvalidFunctionError(msg) from e

    if problem.fun_and_jac is not None:
        try:
            first_crit_and_deriv_eval = problem.fun_and_jac(problem.params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating criterion_and_derivative at start params."
            raise InvalidFunctionError(msg) from e

    if problem.jac is not None:
        used_deriv = first_deriv_eval
    elif problem.fun_and_jac is not None:
        used_deriv = first_crit_and_deriv_eval[1]
    else:
        used_deriv = None

    # ==================================================================================
    # Get the converter (for tree flattening, constraints and scaling)
    # ==================================================================================
    converter, internal_params = get_converter(
        params=problem.params,
        constraints=constraints,
        bounds=problem.bounds,
        func_eval=first_crit_eval.value,
        solver_type=problem.algorithm.algo_info.solver_type,
        scaling=problem.scaling,
        derivative_eval=used_deriv,
        add_soft_bounds=problem.multistart is not None,
    )

    # ==================================================================================
    # initialize the log database
    # ==================================================================================
    logger: LogStore[Any, Any] | None

    if problem.logging:
        logger = LogStore.from_options(problem.logging)
        problem_data = ProblemInitialization(problem.direction, problem.params)
        logger.problem_store.insert(problem_data)
    else:
        logger = None

    # ==================================================================================
    # Do some things that require internal parameters or bounds
    # ==================================================================================

    if converter.has_transforming_constraints and problem.multistart is not None:
        raise NotImplementedError(
            "multistart optimizations are not yet compatible with transforming "
            "constraints."
        )

    # get error penalty function
    error_penalty_func = get_error_penalty_function(
        start_x=internal_params.values,
        start_criterion=first_crit_eval,
        error_penalty=problem.error_penalty,
        solver_type=problem.algorithm.algo_info.solver_type,
        direction=problem.direction,
    )

    # process nonlinear constraints:
    internal_nonlinear_constraints = process_nonlinear_constraints(
        nonlinear_constraints=nonlinear_constraints,
        params=problem.params,
        bounds=problem.bounds,
        converter=converter,
        numdiff_options=problem.numdiff_options,
        skip_checks=problem.skip_checks,
    )

    x = internal_params.values
    internal_bounds = InternalBounds(
        lower=internal_params.lower_bounds,
        upper=internal_params.upper_bounds,
    )
    # ==================================================================================
    # Create a batch evaluator
    # ==================================================================================
    # TODO: Make batch evaluator an argument of maximize and minimize and move this
    # to create_optimization_problem
    batch_evaluator = process_batch_evaluator("joblib")

    # ==================================================================================
    # Create the InternalOptimizationProblem
    # ==================================================================================

    internal_problem = InternalOptimizationProblem(
        fun=problem.fun,
        jac=problem.jac,
        fun_and_jac=problem.fun_and_jac,
        converter=converter,
        solver_type=problem.algorithm.algo_info.solver_type,
        direction=problem.direction,
        bounds=internal_bounds,
        numdiff_options=problem.numdiff_options,
        error_handling=problem.error_handling,
        error_penalty_func=error_penalty_func,
        batch_evaluator=batch_evaluator,
        # TODO: Actually pass through linear constraints if possible
        linear_constraints=None,
        nonlinear_constraints=internal_nonlinear_constraints,
        logger=logger,
    )

    # ==================================================================================
    # Do actual optimization
    # ==================================================================================
    if problem.multistart is None:
        steps = [{"type": "optimization", "name": "optimization"}]

        # TODO: Actually use the step ids
        step_id = log_scheduled_steps_and_get_ids(  # noqa: F841
            steps=steps,
            logging=logger,
        )[0]

        raw_res = problem.algorithm.solve_internal_problem(internal_problem, x, step_id)

    else:
        multistart_options = get_internal_multistart_options_from_public(
            options=problem.multistart,
            params=problem.params,
            params_to_internal=converter.params_to_internal,
        )

        sampling_bounds = InternalBounds(
            lower=internal_params.soft_lower_bounds,
            upper=internal_params.soft_upper_bounds,
        )

        raw_res = run_multistart_optimization(
            local_algorithm=problem.algorithm,
            internal_problem=internal_problem,
            x=x,
            sampling_bounds=sampling_bounds,
            options=multistart_options,
            logging=logger,
            error_handling=problem.error_handling,
        )

    # ==================================================================================
    # Process the result
    # ==================================================================================

    _scalar_start_criterion = cast(
        float, first_crit_eval.internal_value(AggregationLevel.SCALAR)
    )
    log_reader: LogReader[Any] | None

    extra_fields = ExtraResultFields(
        start_fun=_scalar_start_criterion,
        start_params=problem.params,
        algorithm=problem.algorithm.algo_info.name,
        direction=problem.direction,
        n_free=internal_params.free_mask.sum(),
    )

    if problem.multistart is None:
        res = process_single_result(
            raw_res=raw_res,
            converter=converter,
            solver_type=problem.algorithm.algo_info.solver_type,
            extra_fields=extra_fields,
        )
    else:
        res = process_multistart_result(
            raw_res=raw_res,
            converter=converter,
            solver_type=problem.algorithm.algo_info.solver_type,
            extra_fields=extra_fields,
        )

    if logger is not None:
        assert problem.logging is not None
        log_reader = LogReader.from_options(problem.logging)
    else:
        log_reader = None

    res.logger = log_reader

    return res
