import warnings

import numpy as np

from estimagic.config import IS_NLOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS

if IS_NLOPT_INSTALLED:
    import nlopt


DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def nlopt_bobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):

    """Minimize a scalar function using the BOBYQA algorithm.

    The implementation is derived from the BOBYQA subroutine of M. J. D. Powell.

    The algorithm performs derivative free bound-constrained optimization using
    an iteratively constructed quadratic approximation for the objective function.
    Due to its use of quadratic appoximation, the algorithm may perform poorly
    for objective functions that are not twice-differentiable.

    For details see:
    M. J. D. Powell, "The BOBYQA algorithm for bound constrained optimization
    without derivatives," Department of Applied Mathematics and Theoretical
    Physics, Cambridge England, technical report NA2009/06 (2009).

    ``nlopt_bobyqa`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping.max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
    - stopping_max_iterations (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_BOBYQA,
        algorithm_name="nlopt_bobyqa",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_neldermead(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    Do not call this function directly but pass its name "nlopt_bobyqa" to
    estimagic's maximize or minimize function as `algorithm` argument. Specify
    your desired arguments as a dictionary and pass them as `algo_options` to
    minimize or maximize.

    The basic algorithm is described in:
    J. A. Nelder and R. Mead, "A simplex method for function minimization,"
    The Computer Journal 7, p. 308-313 (1965).

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by moving all new points that
    would lie outside the bounds exactly on the bounds.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if np.isfinite(lower_bounds).any():
        warnings.warn(
            "nlopt_neldermead failed on simple benchmark functions if some but not all "
            "bounds were finite. Add finite bounds for all parameters for more safety."
        )

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_NELDERMEAD,
        algorithm_name="nlopt_neldermead",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_praxis(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using principal-axis method.

    This is a gradient-free local optimizer originally described in:
    Richard Brent, Algorithms for Minimization without Derivatives
    (Prentice-Hall, 1972). (Reprinted by Dover, 2002.). It assumes quadratic
    form of the optimized function and repeatedly updates a set of conjugate
    search directions.

    The algorithm, is not invariant to scaling of the objective function and may
    fail under its certain rank-preserving transformations (e.g., will lead to
    a non-quadratic shape of the objective function).

    The algorithm is not determenistic and it is not possible to achieve
    detereminancy via seed setting.

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by returning infinity (Inf)
    when the constraints are violated. The implementation of bound constraints
    is achieved at the const of significantly reduced speed of convergence.
    In case of bounded constraints, this method is dominated by `nlopt_bobyqa`
    and `nlopt_cobyla`.

    Do not call this function directly but pass its name "nlopt_bobyqa" to
    estimagic's maximize or minimize function as `algorithm` argument. Specify
    your desired arguments as a dictionary and pass them as `algo_options` to
    minimize or maximize.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_PRAXIS,
        algorithm_name="nlopt_praxis",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_cobyla(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the cobyla method.

    The alggorithm is derived from Powell's Constrained Optimization BY Linear
    Approximations (COBYLA) algorithm. It is a derivative-free optimizer with
    nonlinear inequality and equality constrains, described in:

    M. J. D. Powell, "A direct search optimization method that models the
    objective and constraint functions by linear interpolation," in Advances in
    Optimization and Numerical Analysis, eds. S. Gomez and J.-P. Hennart (Kluwer
    Academic: Dordrecht, 1994), p. 51-67

    It constructs successive linear approximations of the objective function and
    constraints via a simplex of n+1 points (in n dimensions), and optimizes these
    approximations in a trust region at each step.

    The the nlopt implementation differs from the original implementation in a
    a few ways:
    - Incorporates all of the NLopt termination criteria.
    - Adds explicit support for bound constraints.
    - Allows the algorithm to increase the trust-reion radius if the predicted
    imptoovement was approximately right and the simplex is satisfactory.
    - Pseudo-randomizes simplex steps in the algorithm, aimproving robustness by
    avoiding accidentally taking steps that don't improve conditioning, preserving
    the deterministic nature of the algorithm.
    - Supports unequal initial-step sizes in the different parameters.

    Do not call this function directly but pass its name "nlopt_bobyqa" to
    estimagic's maximize or minimize function as `algorithm` argument. Specify
    your desired arguments as a dictionary and pass them as `algo_options` to
    minimize or maximize.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_COBYLA,
        algorithm_name="nlopt_cobyla",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_sbplx(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the "Subplex" algorithm.

    The alggorithm is a reimplementation of  Tom Rowan's "Subplex" algorithm.
    See: T. Rowan, "Functional Stability Analysis of Numerical Algorithms",
    Ph.D. thesis, Department of Computer Sciences, University of Texas at
    Austin, 1990.

    Subplex is a variant of Nedler-Mead that uses Nedler-Mead on a sequence of
    subspaces. It is climed to be more efficient and robust than the original
    Nedler-Mead algorithm.

    The difference between this re-implementation and the original algorithm
    of Rowan, is that it explicitly supports bound constraints providing big
    improvement in the case where the optimum lies against one of the constraints.


    Do not call this function directly but pass its name "nlopt_bobyqa" to
    estimagic's maximize or minimize function as `algorithm` argument. Specify
    your desired arguments as a dictionary and pass them as `algo_options` to
    minimize or maximize.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_SBPLX,
        algorithm_name="nlopt_sbplx",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_newuoa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the NEWUOA algorithm.

    The algorithm is derived from the NEWUOA subroutine of M.J.D Powell which
    uses iteratively constructed quadratic approximation of the objctive fucntion
    to perform derivative-free unconstrained optimization. Fore more details see:
    M. J. D. Powell, "The NEWUOA software for unconstrained optimization without
    derivatives," Proc. 40th Workshop on Large Scale Nonlinear Optimization
    (Erice, Italy, 2004).

    The algorithm in `nlopt` has been modified to support bound constraints. If all
    of the bound constraints are infinite, this function calls the `nlopt.LN_NEWUOA`
    optimizers for uncsonstrained optimization. Otherwise, the `nlopt.LN_NEWUOA_BOUND`
    optimizer for constrained problems.

    The original algorithm that solves the quadratic subproblems in a spherical
    trust region via a truncated conjugate-gradient algorithm. Thee `nlopt`
    bound-constrained variant uses te `MMA` algorithm for these subproblems to solve
    them with both bound constraints and a sperical trust region.

    `NEWUOA` requires the dimension n of the parameter space to be `≥ 2`, i.e. the
    implementation does not handle one-dimensional optimization problems.

    Do not call this function directly but pass its name "nlopt_bobyqa" to
    estimagic's maximize or minimize function as `algorithm` argument. Specify
    your desired arguments as a dictionary and pass them as `algo_options` to
    minimize or maximize.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if np.any(np.isfinite(lower_bounds)) or np.any(np.isfinite(upper_bounds)):
        alg_name = "nlopt_newuoa_bound"
        alg = nlopt.LN_NEWUOA_BOUND
    else:
        alg_name = "nlopt_newuoa"
        alg = nlopt.LN_NEWUOA

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=alg,
        algorithm_name=alg_name,
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm,
    algorithm_name,
    *,
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = algorithm_name

    def func(x, grad):
        if grad.size > 0:
            criterion, derivative = criterion_and_derivative(
                x,
                task="criterion_and_derivative",
                algorithm_info=algo_info,
            )
            grad[:] = derivative
        else:
            criterion = criterion_and_derivative(
                x,
                task="criterion",
                algorithm_info=algo_info,
            )
        return criterion

    opt = nlopt.opt(algorithm, x.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if lower_bounds is not None:
        opt.set_lower_bounds(lower_bounds)
    if upper_bounds is not None:
        opt.set_upper_bounds(upper_bounds)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    opt.set_min_objective(func)
    solution_x = opt.optimize(x)
    return _process_nlopt_results(opt, solution_x)


def _process_nlopt_results(nlopt_obj, solution_x):
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_relative_criterion_tolerance or "
            + "convergence_absolute_criterion_tolerance was reached"
        ),
        4: (
            "Optimizer stopped because convergence_relative_params_tolerance or "
            + "convergence_absolute_params_tolerance was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    processed = {
        "solution_x": solution_x,
        "solution_criterion": nlopt_obj.last_optimum_value(),
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": nlopt_obj.get_numevals(),
        "n_derivative_evaluations": None,
        "n_iterations": None,
        "success": nlopt_obj.last_optimize_result() in [1, 2, 3, 4],
        "message": messages[nlopt_obj.last_optimize_result()],
        "reached_convergence_criterion": None,
    }
    return processed
