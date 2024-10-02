"""Do a method of simlated moments estimation."""

import functools
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from pybaum import leaf_names, tree_just_flatten

from estimagic.msm_covs import cov_optimal, cov_robust
from estimagic.msm_sensitivity import (
    calculate_actual_sensitivity_to_noise,
    calculate_actual_sensitivity_to_removal,
    calculate_fundamental_sensitivity_to_noise,
    calculate_fundamental_sensitivity_to_removal,
    calculate_sensitivity_to_bias,
    calculate_sensitivity_to_weighting,
)
from estimagic.msm_weighting import get_weighting_matrix
from estimagic.shared_covs import (
    FreeParams,
    calculate_ci,
    calculate_estimation_summary,
    calculate_free_estimates,
    calculate_p_values,
    calculate_summary_data_estimation,
    get_derivative_case,
    transform_covariance,
    transform_free_cov_to_cov,
    transform_free_values_to_params_tree,
)
from optimagic import deprecations, mark
from optimagic.deprecations import (
    replace_and_warn_about_deprecated_bounds,
)
from optimagic.differentiation.derivatives import first_derivative
from optimagic.differentiation.numdiff_options import (
    NumdiffPurpose,
    get_default_numdiff_options,
    pre_process_numdiff_options,
)
from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.fun_value import LeastSquaresFunctionValue
from optimagic.optimization.optimize import minimize
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.block_trees import block_tree_to_matrix, matrix_to_block_tree
from optimagic.parameters.bounds import Bounds, pre_process_bounds
from optimagic.parameters.conversion import Converter, get_converter
from optimagic.parameters.space_conversion import InternalParams
from optimagic.parameters.tree_registry import get_registry
from optimagic.shared.check_option_dicts import (
    check_optimization_options,
)
from optimagic.utilities import get_rng, to_pickle


def estimate_msm(
    simulate_moments,
    empirical_moments,
    moments_cov,
    params,
    optimize_options,
    *,
    bounds=None,
    constraints=None,
    logging=None,
    simulate_moments_kwargs=None,
    weights="diagonal",
    jacobian=None,
    jacobian_kwargs=None,
    jacobian_numdiff_options=None,
    # deprecated
    log_options=None,
    lower_bounds=None,
    upper_bounds=None,
    numdiff_options=None,
):
    """Do a method of simulated moments or indirect inference estimation.

    This is a high level interface for our lower level functions for minimization,
    numerical differentiation, inference and sensitivity analysis. It does the full
    workflow for MSM or indirect inference estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each steps
    vial the optional arguments of this functions. If you find it easier to do the
    minimization separately, you can do so and just provide the optimal parameters as
    ``params`` and set ``optimize_options=False``.

    Args:
        simulate_moments (callable): Function that takes params and potentially other
            keyword arguments and returns a pytree with simulated moments. If the
            function returns a dict containing the key ``"simulated_moments"`` we only
            use the value corresponding to that key. Other entries are stored in the
            log database if you use logging.

        empirical_moments (pandas.Series): A pytree with the same structure as the
            result of ``simulate_moments``.
        moments_cov (pandas.DataFrame): A block-pytree containing the covariance
            matrix of the empirical moments. This is typically calculated with
            our ``get_moments_cov`` function.
        params (pytree): A pytree containing the estimated or start parameters of the
            model. If the supplied parameters are estimated parameters, set
            optimize_options to False. Pytrees can be a numpy array, a pandas Series, a
            DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        optimize_options (dict, Algorithm, str or False): Keyword arguments that govern
            the numerical optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for those that can
            be passed explicitly to ``estimate_msm``.  If you pass False as
            ``optimize_options`` you signal that ``params`` are already
            the optimal parameters and no numerical optimization is needed. If you pass
            a str as optimize_options it is used as the ``algorithm`` option.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.
        simulate_moments_kwargs (dict): Additional keyword arguments for
            ``simulate_moments``.
        weights (str): One of "diagonal" (default), "identity" or "optimal".
            Note that "optimal" refers to the asymptotically optimal weighting matrix
            and is often not a good choice due to large finite sample bias.
        constraints (list, dict): List with constraint dictionaries or single dict.
            See :ref:`constraints`.
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.

        log_options (dict): Additional keyword arguments to configure the logging.

            - "fast_logging" (bool):
                A boolean that determines if "unsafe" settings are used to speed up
                write processes to the database. This should only be used for very short
                running criterion functions where the main purpose of the log is a
                monitoring and it would not be catastrophic to get a corrupted
                database in case of a sudden system shutdown. If one evaluation of the
                criterion function (and gradient if applicable) takes more than 100 ms,
                the logging overhead is negligible.
            - "if_table_exists" (str):
                One of "extend", "replace", "raise". What to do if the tables we want to
                write to already exist. Default "extend".
            - "if_database_exists" (str):
                One of "extend", "replace", "raise". What to do if the database we want
                to write to already exists. Default "extend".
        jacobian (callable): A function that take ``params`` and
            potentially other keyword arguments and returns the jacobian of
            simulate_moments with respect to the params.
        jacobian_kwargs (dict): Additional keyword arguments for the jacobian function.
        jacobian_numdiff_options (dict): Keyword arguments for the calculation of
            numerical derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details. Note that by default we increase the
            step_size by a factor of 2 compared to the rule of thumb for optimal
            step sizes. This is because many msm criterion functions are slightly noisy.

        Returns:
            dict: The estimated parameters, standard errors and sensitivity measures
                and covariance matrix of the parameters.

    """
    # ==================================================================================
    # handle deprecations
    # ==================================================================================

    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )

    if numdiff_options is not None:
        deprecations.throw_numdiff_options_deprecated_in_estimate_msm_future_warning()
        if jacobian_numdiff_options is not None:
            jacobian_numdiff_options = numdiff_options

    deprecations.throw_dict_constraints_future_warning_if_required(constraints)

    # ==================================================================================
    # Check and process inputs
    # ==================================================================================

    bounds = pre_process_bounds(bounds)
    # TODO: Replace dict_constraints with constraints, once we deprecate dictionary
    # constraints.
    dict_constraints = deprecations.pre_process_constraints(constraints)
    jacobian_numdiff_options = pre_process_numdiff_options(jacobian_numdiff_options)
    if jacobian_numdiff_options is None:
        jacobian_numdiff_options = get_default_numdiff_options(
            purpose=NumdiffPurpose.ESTIMATE_JACOBIAN
        )

    if weights not in ["diagonal", "optimal", "identity"]:
        raise NotImplementedError("Custom weighting matrices are not yet implemented.")

    is_optimized = optimize_options is False

    if not is_optimized:
        # If optimize_options is not a dictionary and not False, we assume it represents
        # an algorithm. The actual testing of whether it is a valid algorithm is done
        # when `minimize` is called.
        if not isinstance(optimize_options, dict):
            optimize_options = {"algorithm": optimize_options}

        check_optimization_options(
            optimize_options,
            usage="estimate_msm",
            algorithm_mandatory=True,
        )

    jac_case = get_derivative_case(jacobian)

    weights, internal_weights = get_weighting_matrix(
        moments_cov=moments_cov,
        method=weights,
        empirical_moments=empirical_moments,
        return_type="pytree_and_array",
    )

    internal_moments_cov = block_tree_to_matrix(
        moments_cov,
        outer_tree=empirical_moments,
        inner_tree=empirical_moments,
    )

    jacobian_kwargs = {} if jacobian_kwargs is None else jacobian_kwargs
    simulate_moments_kwargs = (
        {} if simulate_moments_kwargs is None else simulate_moments_kwargs
    )

    # ==================================================================================
    # Calculate estimates via minimization (if necessary)
    # ==================================================================================

    if is_optimized:
        estimates = params
        opt_res = None
    else:
        funcs = get_msm_optimization_functions(
            simulate_moments=simulate_moments,
            empirical_moments=empirical_moments,
            weights=weights,
            simulate_moments_kwargs=simulate_moments_kwargs,
            # Always pass None because we do not support closed form jacobians during
            # optimization yet. Otherwise we would get a NotImplementedError
            jacobian=None,
            jacobian_kwargs=jacobian_kwargs,
        )

        opt_res = minimize(
            bounds=bounds,
            constraints=constraints,
            logging=logging,
            log_options=log_options,
            params=params,
            **funcs,  # contains the criterion func and possibly more
            **optimize_options,
        )

        estimates = opt_res.params

    # ==================================================================================
    # do first function evaluations
    # ==================================================================================

    try:
        sim_mom_eval = simulate_moments(estimates, **simulate_moments_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating simulate_moments at estimated params."
        raise InvalidFunctionError(msg) from e

    if callable(jacobian):
        try:
            jacobian_eval = jacobian(estimates, **jacobian_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at estimated params."
            raise InvalidFunctionError(msg) from e

    else:
        jacobian_eval = None

    # ==================================================================================
    # get converter for params and function outputs
    # ==================================================================================

    if isinstance(sim_mom_eval, dict) and "simulated_moments" in sim_mom_eval:
        func_eval = {"contributions": sim_mom_eval["simulated_moments"]}
    else:
        func_eval = {"contributions": sim_mom_eval}

    converter, internal_estimates = get_converter(
        params=estimates,
        constraints=dict_constraints,
        bounds=bounds,
        func_eval=func_eval,
        solver_type="contributions",
        derivative_eval=jacobian_eval,
    )

    # ==================================================================================
    # Calculate internal jacobian
    # ==================================================================================

    if jac_case == "closed-form":
        x = converter.params_to_internal(estimates)
        int_jac = converter.derivative_to_internal(jacobian_eval, x)
    else:

        def func(x):
            params = converter.params_from_internal(x)
            sim_mom = simulate_moments(params, **simulate_moments_kwargs)
            if isinstance(sim_mom, dict) and "simulated_moments" in sim_mom:
                sim_mom = sim_mom["simulated_moments"]
            registry = get_registry(extended=True)
            out = np.array(tree_just_flatten(sim_mom, registry=registry))
            return out

        int_jac = first_derivative(
            func=func,
            params=internal_estimates.values,
            bounds=Bounds(
                lower=internal_estimates.lower_bounds,
                upper=internal_estimates.upper_bounds,
            ),
            error_handling="continue",
            **asdict(jacobian_numdiff_options),
        ).derivative

    # ==================================================================================
    # Calculate external jac (if no constraints and not closed form )
    # ==================================================================================

    if dict_constraints in [None, []] and jacobian_eval is None and int_jac is not None:
        jacobian_eval = matrix_to_block_tree(
            int_jac,
            outer_tree=empirical_moments,
            inner_tree=estimates,
        )

    if jacobian_eval is None:
        _no_jac_reason = (
            "no closed form jacobian was provided and there are constraints"
        )
    else:
        _no_jac_reason = None

    # ==================================================================================
    # Create MomentsResult
    # ==================================================================================

    free_estimates = calculate_free_estimates(estimates, internal_estimates)

    res = MomentsResult(
        _params=estimates,
        _weights=weights,
        _converter=converter,
        _optimize_result=opt_res,
        _internal_weights=internal_weights,
        _internal_moments_cov=internal_moments_cov,
        _internal_jacobian=int_jac,
        _jacobian=jacobian_eval,
        _no_jacobian_reason=_no_jac_reason,
        _empirical_moments=empirical_moments,
        _internal_estimates=internal_estimates,
        _free_estimates=free_estimates,
        _has_constraints=dict_constraints not in [None, []],
    )
    return res


def get_msm_optimization_functions(
    simulate_moments,
    empirical_moments,
    weights,
    *,
    simulate_moments_kwargs=None,
    jacobian=None,
    jacobian_kwargs=None,
):
    """Construct criterion functions and their derivatives for msm estimation.

    Args:
        simulate_moments (callable): Function that takes params and potentially other
            keyworrd arguments and returns simulated moments as a pandas Series.
            Alternatively, the function can return a dict with any number of entries
            as long as one of those entries is "simulated_moments".
        empirical_moments (pandas.Series): A pandas series with the empirical
            equivalents of the simulated moments.
        weights (pytree): The weighting matrix as block pytree.
        simulate_moments_kwargs (dict): Additional keyword arguments for
            ``simulate_moments``.
        jacobian (callable or pandas.DataFrame): A function that take ``params`` and
            potentially other keyword arguments and returns the jacobian of
            simulate_moments with respect to the params. Alternatively you can pass
            a pandas.DataFrame with the jacobian at the optimal parameters. This is
            only possible if you pass ``optimize_options=False``.
        jacobian_kwargs (dict): Additional keyword arguments for jacobian.

    Returns:
        dict: Dictionary containing at least the entry "fun". If enough inputs
            are provided it also contains the entries "jac" and
            "fun_and_jac". All values are functions that take params
            as only argument.

    """
    flat_weights = block_tree_to_matrix(
        weights,
        outer_tree=empirical_moments,
        inner_tree=empirical_moments,
    )

    chol_weights = np.linalg.cholesky(flat_weights)

    registry = get_registry(extended=True)
    flat_emp_mom = tree_just_flatten(empirical_moments, registry=registry)

    _simulate_moments = _partial_kwargs(simulate_moments, simulate_moments_kwargs)
    _jacobian = _partial_kwargs(jacobian, jacobian_kwargs)

    criterion = mark.least_squares(
        functools.partial(
            _msm_criterion,
            simulate_moments=_simulate_moments,
            flat_empirical_moments=flat_emp_mom,
            chol_weights=chol_weights,
            registry=registry,
        )
    )

    out = {"fun": criterion}

    if _jacobian is not None:
        raise NotImplementedError(
            "Closed form jacobians are not yet supported in estimate_msm"
        )

    return out


def _msm_criterion(
    params, simulate_moments, flat_empirical_moments, chol_weights, registry
):
    """Calculate msm criterion given parameters and building blocks."""
    simulated = simulate_moments(params)
    if isinstance(simulated, dict) and "simulated_moments" in simulated:
        simulated = simulated["simulated_moments"]
    if isinstance(simulated, np.ndarray) and simulated.ndim == 1:
        simulated_flat = simulated
    else:
        simulated_flat = np.array(tree_just_flatten(simulated, registry=registry))

    deviations = simulated_flat - flat_empirical_moments
    residuals = deviations @ chol_weights

    return LeastSquaresFunctionValue(value=residuals)


def _partial_kwargs(func, kwargs):
    """Partial keyword arguments into a function.

    In contrast to normal partial this works if kwargs in None. If func is not a
    callable it simply returns None.

    """
    if isinstance(func, Callable):
        if kwargs not in (None, {}):
            out = functools.partial(func, **kwargs)
        else:
            out = func
    else:
        out = None

    return out


@dataclass
class MomentsResult:
    """Method of moments estimation results object."""

    _params: Any
    _internal_estimates: InternalParams
    _free_estimates: FreeParams
    _weights: Any
    _converter: Converter
    _internal_moments_cov: np.ndarray
    _internal_weights: np.ndarray
    _internal_jacobian: np.ndarray
    _empirical_moments: Any
    _has_constraints: bool
    _optimize_result: Union[OptimizeResult, None] = None
    _jacobian: Any = None
    _no_jacobian_reason: Union[str, None] = None
    _cache: Dict = field(default_factory=dict)

    def _get_free_cov(self, method, n_samples, bounds_handling, seed):
        if method not in {"optimal", "robust"}:
            msg = f"Invalid method {method}. method must be in {'optimal', 'robust'}"
            raise ValueError(msg)
        args = (method, n_samples, bounds_handling, seed)
        is_cached = args in self._cache

        if is_cached:
            free_cov = self._cache[args]
        else:
            free_cov = _calculate_free_cov_msm(
                internal_estimates=self._internal_estimates,
                internal_jacobian=self._internal_jacobian,
                internal_moments_cov=self._internal_moments_cov,
                internal_weights=self._internal_weights,
                converter=self._converter,
                method=method,
                n_samples=n_samples,
                bounds_handling=bounds_handling,
                seed=seed,
            )
            if seed is not None:
                self._cache[args] = free_cov
            elif self._converter.has_transforming_constraints:
                msg = (
                    "seed is set to None and constraints are transforming. This leads "
                    "to randomness in the result. To avoid random behavior, choose a "
                    "non-None seed."
                )
                warnings.warn(msg)

        return free_cov

    @property
    def params(self):
        return self._params

    @property
    def optimize_result(self):
        return self._optimize_result

    @property
    def weights(self):
        return self._weights

    @property
    def jacobian(self):
        return self._jacobian

    @cached_property
    def _se(self):
        return self.se()

    @cached_property
    def _cov(self):
        return self.cov()

    @cached_property
    def _summary(self):
        return self.summary()

    @cached_property
    def _ci(self):
        return self.ci()

    @cached_property
    def _p_values(self):
        return self.p_values()

    def se(
        self,
        method="robust",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate standard errors.

        Args:
            method (str): One of "robust", "optimal". Despite the name, "optimal" is
                not recommended in finite samples and "optimal" standard errors are
                only valid if the asymptotically optimal weighting matrix has been
                used. It is only supported because it is needed to calculate
                sensitivity measures.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.


        Returns:
            Any: A pytree with the same structure as params containing standard errors
                for the parameter estimates.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        free_se = np.sqrt(np.diagonal(free_cov))

        se = transform_free_values_to_params_tree(
            values=free_se,
            free_params=self._free_estimates,
            params=self._params,
        )
        return se

    def cov(
        self,
        method="robust",
        n_samples=10_000,
        bounds_handling="clip",
        return_type="pytree",
        seed=None,
    ):
        """Calculate the variance-covariance matrix of the estimated parameters.

        Args:
            method (str): One of "robust", "optimal". Despite the name, "optimal" is
                not recommended in finite samples and "optimal" standard errors are
                only valid if the asymptotically optimal weighting matrix has been
                used. It is only supported because it is needed to calculate
                sensitivity measures.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            return_type (str): One of "pytree", "array" or "dataframe". Default pytree.
                If "array", a 2d numpy array with the covariance is returned. If
                "dataframe", a pandas DataFrame with parameter names in the
                index and columns are returned.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.


        Returns:
            Any: The covariance matrix of the estimated parameters as block-pytree or
                numpy array.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )
        cov = transform_free_cov_to_cov(
            free_cov=free_cov,
            free_params=self._free_estimates,
            params=self._params,
            return_type=return_type,
        )
        return cov

    def summary(
        self,
        method="robust",
        n_samples=10_000,
        ci_level=0.95,
        bounds_handling="clip",
        seed=None,
    ):
        """Create a summary of estimation results.

        Args:
            method (str): One of "robust", "optimal". Despite the name, "optimal" is
                not recommended in finite samples and "optimal" standard errors are
                only valid if the asymptotically optimal weighting matrix has been
                used. It is only supported because it is needed to calculate
                sensitivity measures.
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.

        Returns:
            Any: The estimation summary as pytree of DataFrames.

        """
        summary_data = calculate_summary_data_estimation(
            self,
            free_estimates=self._free_estimates,
            method=method,
            ci_level=ci_level,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )
        summary = calculate_estimation_summary(
            summary_data=summary_data,
            names=self._free_estimates.all_names,
            free_names=self._free_estimates.free_names,
        )
        return summary

    def ci(
        self,
        method="robust",
        n_samples=10_000,
        ci_level=0.95,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate confidence intervals.

        Args:
            method (str): One of "robust", "optimal". Despite the name, "optimal" is
                not recommended in finite samples and "optimal" standard errors are
                only valid if the asymptotically optimal weighting matrix has been
                used. It is only supported because it is needed to calculate
                sensitivity measures.
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.


        Returns:
            Any: Pytree with the same structure as params containing lower bounds of
                confidence intervals.
            Any: Pytree with the same structure as params containing upper bounds of
                confidence intervals.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        free_lower, free_upper = calculate_ci(
            free_values=self._free_estimates.values,
            free_standard_errors=np.sqrt(np.diagonal(free_cov)),
            ci_level=ci_level,
        )

        lower, upper = (
            transform_free_values_to_params_tree(
                values, free_params=self._free_estimates, params=self._params
            )
            for values in (free_lower, free_upper)
        )
        return lower, upper

    def p_values(
        self,
        method="robust",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate p-values.

        Args:
            method (str): One of "robust", "optimal". Despite the name, "optimal" is
                not recommended in finite samples and "optimal" standard errors are
                only valid if the asymptotically optimal weighting matrix has been
                used. It is only supported because it is needed to calculate
                sensitivity measures.
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.

        Returns:
            Any: Pytree with the same structure as params containing p-values.
            Any: Pytree with the same structure as params containing p-values.

        """
        free_cov = self._get_free_cov(
            method=method,
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        free_p_values = calculate_p_values(
            free_values=self._free_estimates.values,
            free_standard_errors=np.sqrt(np.diagonal(free_cov)),
        )

        p_values = transform_free_values_to_params_tree(
            free_p_values, free_params=self._free_estimates, params=self._params
        )
        return p_values

    def sensitivity(
        self,
        kind="bias",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
        return_type="pytree",
    ):
        """Calculate sensitivity measures for moments estimates.

        The sensitivity measures are based on the following papers:

        Andrews, Gentzkow & Shapiro (2017, Quarterly Journal of Economics)

        Honore, Jorgensen & de Paula
        (https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2779)

        In the papers the different kinds of sensitivity measures are just called
        m1, e2, e3, e4, e5 and e6. We try to give them more informative names, but
        list the original names for references.

        Args:
            kind (str): The following kinds are supported:

                - "bias":
                    Origally m1. How strongly would the parameter estimates be biased if
                    the kth moment was misspecified, i.e not zero in expectation?
                - "noise_fundamental":
                    Originally e2. How much precision would be lost if the kth moment
                    was subject to a little additional noise if the optimal weighting
                    matrix was used?
                - "noise":
                    Originally e3. How much precision would be lost if the kth moment
                    was subjet to a little additional noise?
                - "removal":
                    Originally e4. How much precision would be lost if the kth moment
                    was excluded from the estimation?
                - "removal_fundamental":
                    Originally e5. How much precision would be lost if the kth moment
                    was excluded from the estimation if the asymptotically optimal
                    weighting matrix was used.
                - "weighting":
                    Originally e6. How would the precision change if the weight of the
                    kth moment is increased a little?
            n_samples (int): Number of samples used to transform the covariance matrix
                of the internal parameter vector into the covariance matrix of the
                external parameters. For background information about internal and
                external params see :ref:`implementation_of_constraints`. This is only
                used if you are using constraints.
            bounds_handling (str): One of "clip", "raise", "ignore". Determines how
                bounds are handled. If "clip", confidence intervals are clipped at the
                bounds. Standard errors are only adjusted if a sampling step is
                necessary due to additional constraints. If "raise" and any lower or
                upper bound is binding, we raise an Error. If "ignore", boundary
                problems are simply ignored.
            seed (int): Seed for the random number generator. Only used if there are
                transforming constraints.
            return_type (str): One of "array", "dataframe" or "pytree". Default pytree.
                If your params or moments have a very nested format, return_type
                "dataframe" might be the better choice.

        Returns:
            Any: The sensitivity measure as a pytree, numpy array or DataFrame.
                In 2d formats, the sensitivity measures have one row per estimated
                parameter and one column per moment.

        """
        if self._has_constraints:
            raise NotImplementedError(
                "Sensitivity measures with constraints are not yet implemented."
            )
        jac = self._internal_jacobian
        weights = self._internal_weights
        moments_cov = self._internal_moments_cov
        params_cov = self._get_free_cov(
            method="robust",
            n_samples=n_samples,
            bounds_handling=bounds_handling,
            seed=seed,
        )

        weights_opt = get_weighting_matrix(
            moments_cov=moments_cov,
            method="optimal",
            empirical_moments=self._empirical_moments,
        )
        params_cov_opt = cov_optimal(jac, weights_opt)

        if kind == "bias":
            raw = calculate_sensitivity_to_bias(jac=jac, weights=weights)
        elif kind == "noise_fundamental":
            raw = calculate_fundamental_sensitivity_to_noise(
                jac=jac,
                weights=weights_opt,
                moments_cov=moments_cov,
                params_cov_opt=params_cov_opt,
            )
        elif kind == "noise":
            m1 = calculate_sensitivity_to_bias(jac=jac, weights=weights)
            raw = calculate_actual_sensitivity_to_noise(
                sensitivity_to_bias=m1,
                weights=weights,
                moments_cov=moments_cov,
                params_cov=params_cov,
            )
        elif kind == "removal":
            raw = calculate_actual_sensitivity_to_removal(
                jac=jac,
                weights=weights,
                moments_cov=moments_cov,
                params_cov=params_cov,
            )
        elif kind == "removal_fundamental":
            raw = calculate_fundamental_sensitivity_to_removal(
                jac=jac,
                moments_cov=moments_cov,
                params_cov_opt=params_cov_opt,
            )

        elif kind == "weighting":
            raw = calculate_sensitivity_to_weighting(
                jac=jac,
                weights=weights,
                moments_cov=moments_cov,
                params_cov=params_cov,
            )
        else:
            raise ValueError(f"Invalid kind: {kind}")

        if return_type == "array":
            out = raw
        elif return_type == "pytree":
            out = matrix_to_block_tree(
                raw,
                outer_tree=self._params,
                inner_tree=self._empirical_moments,
            )
        elif return_type == "dataframe":
            registry = get_registry(extended=True)
            row_names = self._internal_estimates.names
            col_names = leaf_names(self._empirical_moments, registry=registry)
            out = pd.DataFrame(
                data=raw,
                index=row_names,
                columns=col_names,
            )
        else:
            msg = (
                f"Invalid return type: {return_type}. Valid are 'pytree', 'array' "
                "and 'dataframe'"
            )
            raise ValueError(msg)
        return out

    def to_pickle(self, path):
        """Save the MomentsResult object to pickle.

        Args:
            path (str, pathlib.Path): A str or pathlib.path ending in .pkl or .pickle.

        """
        to_pickle(self, path=path)


def _calculate_free_cov_msm(
    internal_estimates,
    internal_jacobian,
    internal_moments_cov,
    internal_weights,
    converter,
    method,
    n_samples,
    bounds_handling,
    seed,
):
    if method == "optimal":
        internal_cov = cov_optimal(internal_jacobian, internal_weights)
    else:
        internal_cov = cov_robust(
            internal_jacobian, internal_weights, internal_moments_cov
        )

    rng = get_rng(seed)

    free_cov = transform_covariance(
        internal_params=internal_estimates,
        internal_cov=internal_cov,
        converter=converter,
        n_samples=n_samples,
        rng=rng,
        bounds_handling=bounds_handling,
    )
    return free_cov
