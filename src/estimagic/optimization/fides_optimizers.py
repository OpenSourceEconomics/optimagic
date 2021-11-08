"""Implement the fides optimizer."""
import logging
from functools import partial

import numpy as np
from estimagic.config import IS_FIDES_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS

if IS_FIDES_INSTALLED:
    from fides import hessian_approximation
    from fides import Optimizer


def fides(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    hessian_update_strategy="bfgs",
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_absolute_gradient_tolerance=CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE,
    convergence_relative_gradient_tolerance=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    stopping_max_seconds=np.inf,
    trustregion_initial_radius=1.0,
    trustregion_stepback_strategy="reflect",
    trustregion_subspace_dimension="2D",
    trustregion_max_stepback_fraction=0.95,
    trustregion_decrease_threshold=0.25,
    trustregion_increase_threshold=0.75,
    trustregion_decrease_factor=0.25,
    trustregion_increase_factor=2.0,
    trustregion_refine_stepback=False,
    trustregion_scaled_gradient_as_possible_stepback=False,
):
    """Minimize a scalar function using the Fides Optimizer.

    - hessian_update_strategy (str): Hessian Update Strategy to employ. You can provide
      a lowercase or uppercase string or a
      fides.hession_approximation.HessianApproximation class instance. FX, SSM, TSSM and
      GNSBFGS are not supported by estimagic. The available update strategies are:

        - **bb**: Broydens "bad" method as introduced :cite:`Broyden1965`.
        - **bfgs**: Broyden-Fletcher-Goldfarb-Shanno update strategy.
        - **bg**: Broydens "good" method as introduced in :cite:`Broyden1965`.
        - You can use a general BroydenClass Update scheme using the Broyden class from
          `fides.hessian_approximation`. This is a generalization of BFGS/DFP methods
          where the parameter :math:`phi` controls the convex combination between the
          two. This is a rank 2 update strategy that preserves positive-semidefiniteness
          and symmetry (if :math:`\\phi \\in [0,1]`). It is described in
          :cite:`Nocedal1999`, Chapter 6.3.
        - **dfp**: Davidon-Fletcher-Powell update strategy.
        - **sr1**: Symmetric Rank 1 update strategy as described in :cite:`Nocedal1999`,
          Chapter 6.2.

      Or you can pass a class instance directly. See the `Fides' Documentation
      <https://fides-optimizer.readthedocs.io/en/latest/generated/fides.hessian_approximation.html>`_
      for more details.

    - convergence.absolute_criterion_tolerance (float): absolute convergence criterion
      tolerance. This is only the interpretation of this parameter if the relative
      criterion tolerance is set to 0. Denoting the absolute criterion tolerance by
      :math:`\alpha` and the relative criterion tolerance by :math:`\beta`, the
      convergence condition on the criterion improvement is :math:`|f(x_k) - f(x_{k-1})|
      < \\alpha + \\beta \\cdot |f(x_{k-1})|`
    - convergence.relative_criterion_tolerance (float): relative convergence criterion
      tolerance. This is only the interpretation of this parameter if the absolute
      criterion tolerance is set to 0 (as is the default). Denoting the absolute
      criterion tolerance by :math:`\alpha` and the relative criterion tolerance by
      :math:`\beta`, the convergence condition on the criterion improvement is
      :math:`|f(x_k) - f(x_{k-1})| < \\alpha + \\beta \\cdot |f(x_{k-1})|`
    - convergence.absolute_params_tolerance (float): The optimization terminates
      successfully when the step size falls below this number, i.e. when
      :math:`||x_{k+1} - x_k||` is smaller than this tolerance.
    - convergence.absolute_gradient_tolerance (float): The optimization terminates
      successfully when the gradient norm is less or equal than this tolerance.
    - convergence.relative_gradient_tolerance (float): The optimization terminates
      successfully when the norm of the gradient divided by the absolute function value
      is less or equal to this tolerance.

    - stopping.max_iterations (int): maximum number of allowed iterations.
    - stopping.max_seconds (int): maximum number of walltime seconds, deactivated by
      default.

    - trustregion.initial_radius (float): Initial trust region radius. Default is 1.
    - trustregion.stepback_strategy (str): search refinement strategy if proposed step
      reaches a parameter bound. The default is "reflect". The available options are:

        - "reflect": recursive reflections at boundary.
        - "reflect_single": single reflection at boundary.
        - "truncate": truncate step at boundary and re-solve the restricted subproblem
        - "mixed": mix reflections and truncations

    - trustregion.subspace_dimension (str): Subspace dimension in which the subproblem
      will be solved. The default is "2D". The following values are available:

        - "2D": Two dimensional Newton/Gradient subspace
        - "full": full dimensionality
        - "scg": Conjugated Gradient subspace via Steihaug's method

    - trustregion.max_stepback_fraction (float): Stepback parameter that controls how
      close steps are allowed to get to the boundary. It is the maximal fraction of a
      step to take if full step would reach breakpoint.

    - trustregion.decrease_threshold (float): Acceptance threshold for trust region
      ratio. The default is 0.25 (:cite:`Nocedal2006`). The radius is decreased if the
      trust region ratio is below this value. This is denoted by :math:`\\mu` in
      algorithm 4.1 in :cite:`Nocedal2006`.
    - trustregion.increase_threshold (float): Threshold for the trust region radius
      ratio above which the trust region radius can be increased. This is denoted by
      :math:`\\eta` in algorithm 4.1 in :cite:`Nocedal2006`. The default is 0.75
      (:cite:`Nocedal2006`).
    - trustregion.decrease_factor (float): factor by which trust region radius will be
      decreased in case it is decreased. This is denoted by :math:`\\gamma_1` in
      algorithm 4.1 in :cite:`Nocedal2006` and its default is 0.25.
    - trustregion.increase_factor (float): factor by which trust region radius will be
      increase in case it is increase. This is denoted by :math:`\\gamma_2` in algorithm
      4.1 in :cite:`Nocedal2006` and its default is 2.0.

    - trustregion.refine_stepback (bool): whether to refine stepbacks via optimization.
      Default is False.
    - trustregion.scaled_gradient_as_possible_stepback (bool): whether the scaled
      gradient should be added to the set of possible stepback proposals. Default is
      False.

    Returns: dict: See :ref:`internal_optimizer_output` for details.

    """
    if not IS_FIDES_INSTALLED:
        raise NotImplementedError(
            "The fides package is not installed. You can install it with "
            "`pip install fides>=0.6.3`."
        )

    fides_options = {
        "fatol": convergence_absolute_criterion_tolerance,
        "frtol": convergence_relative_criterion_tolerance,
        "xtol": convergence_absolute_params_tolerance,
        "gatol": convergence_absolute_gradient_tolerance,
        "grtol": convergence_relative_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "delta_init": trustregion_initial_radius,
        "maxtime": stopping_max_seconds,
        "stepback_strategy": trustregion_stepback_strategy,
        "subspace_solver": trustregion_subspace_dimension,
        "theta_max": trustregion_max_stepback_fraction,
        "mu": trustregion_decrease_threshold,
        "eta": trustregion_increase_threshold,
        "gamma1": trustregion_decrease_factor,
        "gamma2": trustregion_increase_factor,
        "refine_stepback": trustregion_refine_stepback,
        "scaled_gradient": trustregion_scaled_gradient_as_possible_stepback,
    }

    algo_info = {
        "primary_criterion_entry": "value",
        "parallelizes": False,
        "needs_scaling": False,
        "name": "fides",
    }

    fun = partial(
        criterion_and_derivative,
        task="criterion_and_derivative",
        algorithm_info=algo_info,
    )

    hessian_instance = _create_hessian_updater_from_user_input(hessian_update_strategy)

    opt = Optimizer(
        fun=fun,
        lb=lower_bounds,
        ub=upper_bounds,
        hessian_update=hessian_instance,
        verbose=logging.ERROR,
        resfun=False,
        options=fides_options,
    )
    raw_res = opt.minimize(x)
    res = _process_fides_res(raw_res, opt)
    return res


def _process_fides_res(raw_res, opt):
    """Create an estimagic results dictionary from the Fides output.

    Args:
        raw_res (tuple): Tuple containing the Fides result
        opt (fides.Optimizer): Fides Optimizer after minimize has been called on it.

    """
    fval, x, grad, hess = raw_res
    res = {
        "solution_criterion": fval,
        "solution_x": x,
        "solution_derivative": grad,
        "solution_hessian": hess,
        "success": opt.converged,
        "n_iterations": opt.iteration,
    }
    return res


def _create_hessian_updater_from_user_input(hessian_update_strategy):
    hessians_needing_residuals = (
        hessian_approximation.FX,
        hessian_approximation.SSM,
        hessian_approximation.TSSM,
        hessian_approximation.GNSBFGS,
    )
    unsupported_hess_msg = (
        f"{hessian_update_strategy} not supported because it requires "
        + "residuals. Choose one of 'BB', 'BFGS', 'BG', 'DFP' or 'SR1' or pass "
        + "an instance of the fides.hessian_approximation.HessianApproximation "
        + "class."
    )

    if hessian_update_strategy in ("broyden", "Broyden", "BROYDEN"):
        raise ValueError(
            "You cannot use the Broyden update strategy without specifying the "
            "interpolation parameter phi. Import the Broyden class from "
            "`fides.hessian_approximation`, create an instance of it with your "
            "desired value of phi and pass this instance instead."
        )
    elif isinstance(hessian_update_strategy, str):
        if hessian_update_strategy.lower() in ["fx", "ssm", "tssm", "gnsbfgs"]:
            raise NotImplementedError(unsupported_hess_msg)
        else:
            hessian_name = hessian_update_strategy.upper()
            hessian_class = getattr(hessian_approximation, hessian_name)
            hessian_instance = hessian_class()
    elif isinstance(
        hessian_update_strategy, hessian_approximation.HessianApproximation
    ):
        hessian_instance = hessian_update_strategy
        if isinstance(hessian_instance, hessians_needing_residuals):
            raise NotImplementedError(unsupported_hess_msg)
    else:
        raise ValueError(
            "You must provide a hessian_update_strategy that is either a string or an "
            "instance of the fides.hessian_approximation.HessianApproximation class."
        )
    return hessian_instance