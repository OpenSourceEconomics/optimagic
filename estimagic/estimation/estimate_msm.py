"""Do a method of simlated moments estimation."""
import functools
from collections.abc import Callable

import numpy as np
import pandas as pd

from estimagic.estimation.msm_weighting import get_weighting_matrix
from estimagic.inference.msm_covs import cov_efficient
from estimagic.inference.msm_covs import cov_sandwich
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import get_internal_first_derivative
from estimagic.inference.shared import transform_covariance
from estimagic.optimization.optimize import minimize
from estimagic.parameters.process_constraints import process_constraints


def estimate_msm(
    simulate_moments,
    empirical_moments,
    moments_cov,
    params,
    minimize_options,
    *,
    constraints=None,
    logging=False,
    log_options=None,
    simulate_moments_kwargs=None,
    weights="diagonal",
    numdiff_options=None,
    jacobian=None,
    jacobian_kwargs=None,
    simulate_moments_and_jacobian=None,
    simulate_moments_and_jacobian_kwargs=None,
    ci_level=0.95,  # noqa: U100
    n_samples=10_000,
    bounds_handling="raise",
):
    """Do a method of simulated moments or indirect inference estimation.

    This is a high level interface for our lower level functions for minimization,
    numerical differentiation, inference and sensitivity analysis. It does the full
    workflow for MSM or indirect inference estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each steps
    vial the optional arguments of this functions. If you find it easier to do the
    "difficult" steps (mainly minimization and calculating numerical derivatives
    of a potentially noisy function) separately, you can do so and just provide those
    results as ``params`` and ``jacobian``.

    The docstring is aspirational and not all options are supported yet.

    Args:
        simulate_moments (callable): Function that takes params and potentially other
            keyworrd arguments and returns simulated moments as a pandas Series.
            Alternatively, the function can return a dict with any number of entries
            as long as one of those entries is "simulated_moments".
        empirical_moments (pandas.Series): A pandas series with the empirical
            equivalents of the simulated moments.
        moments_cov (pandas.DataFrame): A quadratic pandas DataFrame with the covariance
            matrix of the empirical moments. This is typically calculated with
            our ``get_moments_cov`` function. The index and columns need to be the same
            as the index of ``empirical_moments``.
        params (pandas.DataFrame): Start params for the optimization. See :ref:`params`
            for details.
        simulate_moments_kwargs (dict): Additional keyword arguments for
            ``simulate_moments``.
        weights (str or pandas.DataFrame): Either a DataFrame with a positive
            semi-definite weighting matrix or a string that specifies one of the
            pre-implemented weighting matrices: "diagonal" (default), "identity" or
            "optimal". Note that "optimal" refers to the asymptotically optimal
            weighting matrix and is often not a good choice due to large finite sample
            bias.
        constraints (list): List with constraint dictionaries.
            See .. _link: ../../docs/source/how_to_guides/how_to_use_constraints.ipynb
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
            When doing parallel optimizations and logging is provided, you have to
            provide a different path for each optimization you are running. You can
            disable logging completely by setting it to False, but we highly recommend
            not to do so. The dashboard can only be used when logging is used.
        log_options (dict): Additional keyword arguments to configure the logging.
            - "suffix": A string that is appended to the default table names, separated
            by an underscore. You can use this if you want to write the log into an
            existing database where the default names "optimization_iterations",
            "optimization_status" and "optimization_problem" are already in use.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is a real-time dashboard and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_exists": (str) One of "extend", "replace", "raise"
            - "save_all_arguments": (bool). If True, all arguments to maximize
              that can be pickled are saved in the log file. Otherwise, only the
              information needed by the dashboard is saved. Default False.
        minimize_options (dict or False): Keyword arguments that govern the numerical
            optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for criterion,
            derivative, criterion_and_derivative and params. If you pass False as
            minimize_options you signal that ``params`` are already the optimal
            parameters and no numerical optimization is needed.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        jacobian (callable or pandas.DataFrame): A function that take ``params`` and
            potentially other keyword arguments and returns the jacobian of
            simulate_moments with respect to the params. Alternatively you can pass
            a pandas.DataFrame with the jacobian at the optimal parameters. This is
            only possible if you pass ``minimize_options=False``.
        jacobian_kwargs (dict): Additional keyword arguments for jacobian.
        simulate_moments_and_jacobian (callable): A function that takes params and
            potentially other keyword arguments and returns a tuple with simulated
            moments and the jacobian of simulated moments with respect to params.
        simulate_moments_and_jacobian_kwargs (dict): Additional keyword arguments for
            simulate_moments_and_jacobian.
        ci_level (float): Confidence level for the calculation of confidence intervals.
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters. For background information about internal and external params
            see :ref:`implementation_of_constraints`. This is only used if you have
            constraints in the ``minimize_options``
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.

        Returns:
            dict: The estimated parameters, standard errors and sensitivity measures.

    """
    is_minimized = minimize_options is False
    is_differentiated = isinstance(jacobian, (pd.DataFrame, np.ndarray))
    needs_numdiff = jacobian is None
    is_optimal_weights = weights == "optimal"

    if not isinstance(weights, (np.ndarray, pd.DataFrame)):
        weights = get_weighting_matrix(moments_cov, weights)

    if (not is_minimized) and is_differentiated:
        raise ValueError(
            "Providing a pre-calculated jacobian is only possible if the minimization "
            "was done outside of estimate_msm, i.e. if minimize_options=False."
        )

    if not isinstance(minimize_options, dict) or "algorithm" not in minimize_options:
        raise ValueError(
            "minimize_options must be a dict containing at least the entry 'algorithm'"
        )

    numdiff_options = numdiff_options if numdiff_options is not None else {}
    constraints = [] if constraints is None else constraints

    if is_minimized:
        min_res = {"solution_params": params}
    else:
        funcs = get_msm_optimization_functions(
            simulate_moments=simulate_moments,
            empirical_moments=empirical_moments,
            weights=weights,
            simulate_moments_kwargs=simulate_moments_kwargs,
            jacobian=jacobian,
            jacobian_kwargs=jacobian_kwargs,
            simulate_moments_and_jacobian=simulate_moments_and_jacobian,
            simulate_moments_and_jacobian_kwargs=simulate_moments_and_jacobian_kwargs,
        )
        # order ensures that invalid entries of minimize options are overwritten
        min_kwargs = {
            "constraints": constraints,
            "logging": logging,
            "log_options": log_options,
            "params": params,
            **funcs,
        }
        if minimize_options is not None:
            min_kwargs = {**minimize_options, **min_kwargs}

        min_res = minimize(**min_kwargs)

    estimates = min_res["solution_params"]

    if is_differentiated:
        jac = jacobian
    elif isinstance(jacobian, Callable):
        jacobian_kwargs = {} if jacobian_kwargs is None else jacobian_kwargs
        jac = jacobian(estimates, **jacobian_kwargs)
        if constraints is not None:
            raise NotImplementedError(
                "Closed form jacobian is not yet compatible with constraints."
            )
    else:
        deriv_res = get_internal_first_derivative(
            func=simulate_moments,
            params=estimates,
            constraints=constraints,
            func_kwargs=simulate_moments_kwargs,
            numdiff_options={**numdiff_options, "key": "simulated_moments"},
        )
        jac = deriv_res["derivative"]
        numdiff_info = {k: v for k, v in deriv_res.items() if k != "derivative"}

    if is_optimal_weights:
        cov = cov_efficient(jac, weights)
    else:
        cov = cov_sandwich(jac, weights, moments_cov)

    cov = transform_covariance(
        params=params,
        internal_cov=cov,
        constraints=constraints,
        n_samples=n_samples,
        bounds_handling=bounds_handling,
    )

    summary = calculate_inference_quantities(
        params=min_res["solution_params"],
        free_cov=cov,
    )

    out = {"summary": summary, "cov": cov}

    if not is_minimized:
        out["minimize_res"] = min_res

    if needs_numdiff:
        out["numdiff_info"] = numdiff_info

    processed_constraints, _ = process_constraints(constraints, params)

    if processed_constraints:
        out["jacobian"] = "No external jacobian defined due to constraints."
    else:
        out["jacobian"] = pd.DataFrame(
            jac, columns=cov.index, index=moments_cov.columns
        )

    return out


def get_msm_optimization_functions(
    simulate_moments,
    empirical_moments,
    weights,
    simulate_moments_kwargs=None,
    jacobian=None,
    jacobian_kwargs=None,
    simulate_moments_and_jacobian=None,
    simulate_moments_and_jacobian_kwargs=None,
):
    """Construct criterion functions and their derivatives for msm estimation.

    Args:


    Returns:
        dict: Dictionary containing at least the entry "criterion". If enough inputs
            are provided it also contains the entries "derivative" and
            "criterion_and_derivative". All values are functions that take params
            as only argument.

    """
    _simulate_moments = _partial_kwargs(simulate_moments, simulate_moments_kwargs)
    _jacobian = _partial_kwargs(jacobian, jacobian_kwargs)
    _simulate_moments_and_jacobian = _partial_kwargs(
        simulate_moments_and_jacobian, simulate_moments_and_jacobian_kwargs
    )

    criterion = functools.partial(
        _msm_criterion,
        simulate_moments=_simulate_moments,
        empirical_moments=empirical_moments,
        weights=weights,
    )

    out = {"criterion": criterion}

    if _jacobian is not None:
        raise NotImplementedError(
            "Closed form jacobians are not yet supported in estimate_msm"
        )

    if _simulate_moments_and_jacobian is not None:
        raise NotImplementedError(
            "Closed form jacobians are not yet supported in estimate_msm"
        )

    return out


def _msm_criterion(params, simulate_moments, empirical_moments, weights):
    simulated = simulate_moments(params)
    deviations = simulated - empirical_moments
    out = deviations @ weights @ deviations
    return out


def _partial_kwargs(func, kwargs):
    if isinstance(func, Callable):
        if kwargs not in (None, {}):
            out = functools.partial(func, **kwargs)
        else:
            out = func
    else:
        out = None

    return out
