"""Public functions for optimization.

This module defines the public functions `maximize` and `minimize` that will be called
by users.

Internally, `maximize` and `minimize` just call `create_optimization_problem` with
all arguments and add the `direction`. In `create_optimization_problem`, the user input
is consolidated and converted to stricter types.  The resulting `OptimizationProblem`
is then passed to `_optimize` which handles the optimization logic.

`_optimize` processes the optimization problem and performs the actual optimization.

"""

import warnings
from pathlib import Path
from typing import cast

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.exceptions import (
    InvalidFunctionError,
)
from optimagic.logging.create_tables import (
    make_optimization_iteration_table,
    make_optimization_problem_table,
    make_steps_table,
)
from optimagic.logging.load_database import load_database
from optimagic.logging.write_to_database import append_row
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
from optimagic.typing import AggregationLevel, Direction, ErrorHandling


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
    logging=False,
    log_options=None,
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
    logging=False,
    log_options=None,
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


def _optimize(problem: OptimizationProblem) -> OptimizeResult:
    """Solve an optimization problem."""
    # ==================================================================================
    # Split constraints into nonlinear and reparametrization parts
    # ==================================================================================
    constraints = problem.constraints
    if isinstance(constraints, dict):
        constraints = [constraints]

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
    if problem.logging:
        # TODO: We want to remove the optimization_problem table completely but we
        # probably do need to store the start parameters in the database because it is
        # used by the log reader.
        problem_data = {
            "direction": problem.direction.value,
            "params": problem.params,
        }
        database = _create_and_initialize_database(
            logging=problem.logging,
            log_options=problem.log_options,
            problem_data=problem_data,
        )
    else:
        database = None

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
        error_handling=problem.error_handling,
        start_x=internal_params.values,
        start_criterion=first_crit_eval,
        error_penalty=problem.error_penalty,
        solver_type=problem.algorithm.algo_info.solver_type,
        direction=problem.direction,
    )

    # convert the error handling to an enum
    # TODO: Admit enums in the outer interface and do this processing in
    # create_optimization_problem
    if problem.error_handling == "raise":
        internal_error_handling = ErrorHandling.RAISE
    else:
        internal_error_handling = ErrorHandling.CONTINUE

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
        error_handling=internal_error_handling,
        error_penalty_func=error_penalty_func,
        batch_evaluator=batch_evaluator,
        # TODO: Actually pass through linear constraints if possible
        linear_constraints=None,
        nonlinear_constraints=internal_nonlinear_constraints,
        # TODO: add logger
    )

    # ==================================================================================
    # Do actual optimization
    # ==================================================================================
    if problem.multistart is None:
        steps = [{"type": "optimization", "name": "optimization"}]

        # TODO: Actually use the step ids
        step_ids = log_scheduled_steps_and_get_ids(  # noqa: F841
            steps=steps,
            logging=problem.logging,
            database=database,
        )

        raw_res = problem.algorithm.solve_internal_problem(internal_problem, x)

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
            logging=problem.logging,
            database=database,
            error_handling=problem.error_handling,
        )

    # ==================================================================================
    # Process the result
    # ==================================================================================

    _scalar_start_criterion = cast(
        float, first_crit_eval.internal_value(AggregationLevel.SCALAR)
    )

    extra_fields = ExtraResultFields(
        start_fun=_scalar_start_criterion,
        start_params=problem.params,
        algorithm=problem.algorithm.algo_info.name,
        direction=problem.direction,
        n_free=internal_params.free_mask.sum(),
    )

    if problem.multistart is None:
        res = process_single_result(
            res=raw_res,
            converter=converter,
            solver_type=problem.algorithm.algo_info.solver_type,
            extra_fields=extra_fields,
        )
    else:
        res = process_multistart_result(
            res=raw_res,
            converter=converter,
            solver_type=problem.algorithm.algo_info.solver_type,
            extra_fields=extra_fields,
        )

    return res


def _create_and_initialize_database(logging, log_options, problem_data):
    """Create and initialize to sqlite database for logging."""
    path = Path(logging)
    fast_logging = log_options.get("fast_logging", False)
    if_table_exists = log_options.get("if_table_exists", "extend")
    if_database_exists = log_options.get("if_database_exists", "extend")

    if "if_exists" in log_options and "if_table_exists" not in log_options:
        warnings.warn("The log_option 'if_exists' was renamed to 'if_table_exists'.")

    if logging.exists():
        if if_database_exists == "raise":
            raise FileExistsError(
                f"The database {logging} already exists and the log_option "
                "'if_database_exists' is set to 'raise'"
            )
        elif if_database_exists == "replace":
            logging.unlink()

    database = load_database(path_or_database=path, fast_logging=fast_logging)

    # create the optimization_iterations table
    make_optimization_iteration_table(
        database=database,
        if_exists=if_table_exists,
    )

    # create and initialize the steps table; This is alway extended if it exists.
    make_steps_table(database, if_exists=if_table_exists)

    # create_and_initialize the optimization_problem table
    make_optimization_problem_table(database, if_exists=if_table_exists)

    not_saved = [
        "criterion",
        "criterion_kwargs",
        "constraints",
        "derivative",
        "derivative_kwargs",
        "criterion_and_derivative",
        "criterion_and_derivative_kwargs",
    ]
    problem_data = {
        key: val for key, val in problem_data.items() if key not in not_saved
    }

    append_row(problem_data, "optimization_problem", database=database)

    return database
