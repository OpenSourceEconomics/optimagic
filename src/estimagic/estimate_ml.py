import warnings
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict

import numpy as np
import pandas as pd

from estimagic.ml_covs import (
    cov_cluster_robust,
    cov_hessian,
    cov_jacobian,
    cov_robust,
    cov_strata_robust,
)
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
from optimagic.differentiation.derivatives import first_derivative, second_derivative
from optimagic.differentiation.numdiff_options import (
    NumdiffPurpose,
    get_default_numdiff_options,
    pre_process_numdiff_options,
)
from optimagic.exceptions import InvalidFunctionError, NotAvailableError
from optimagic.optimization.fun_value import (
    convert_fun_output_to_function_value,
    enforce_return_type,
)
from optimagic.optimization.optimize import maximize
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.block_trees import block_tree_to_matrix, matrix_to_block_tree
from optimagic.parameters.bounds import Bounds, pre_process_bounds
from optimagic.parameters.conversion import Converter, get_converter
from optimagic.parameters.space_conversion import InternalParams
from optimagic.shared.check_option_dicts import (
    check_optimization_options,
)
from optimagic.typing import AggregationLevel
from optimagic.utilities import get_rng, to_pickle


def estimate_ml(
    loglike,
    params,
    optimize_options,
    *,
    bounds=None,
    constraints=None,
    logging=None,
    loglike_kwargs=None,
    jacobian=None,
    jacobian_kwargs=None,
    jacobian_numdiff_options=None,
    hessian=None,
    hessian_kwargs=None,
    hessian_numdiff_options=None,
    design_info=None,
    # deprecated
    log_options=None,
    lower_bounds=None,
    upper_bounds=None,
    numdiff_options=None,
):
    """Do a maximum likelihood (ml) estimation.

    This is a high level interface of our lower level functions for maximization,
    numerical differentiation and inference. It does the full workflow for maximum
    likelihood estimation with just one function call.

    While we have good defaults, you can still configure each aspect of each step
    via the optional arguments of this function. If you find it easier to do the
    maximization separately, you can do so and just provide the optimal parameters as
    ``params`` and set ``optimize_options=False``

    Args:
        loglike (callable): Likelihood function that takes a params (and potentially
            other keyword arguments) a pytree containing the likelihood contributions
            for each observation or a FunctionValue object.
        params (pytree): A pytree containing the estimated or start parameters of the
            likelihood model. If the supplied parameters are estimated parameters, set
            optimize_options to False. Pytrees can be a numpy array, a pandas Series, a
            DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        optimize_options (dict, Algorithm, str or False): Keyword arguments that govern
            the numerical optimization. Valid entries are all arguments of
            :func:`~estimagic.optimization.optimize.minimize` except for those that are
            passed explicilty to ``estimate_ml``. If you pass False as optimize_options
            you signal that ``params`` are already the optimal parameters and no
            numerical optimization is needed. If you pass a str as optimize_options it
            is used as the ``algorithm`` option.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.
        constraints (list, dict): List with constraint dictionaries or single dict.
            See :ref:`constraints`.
        logging (pathlib.Path, str or False): Path to sqlite3 file (which typically has
            the file extension ``.db``. If the file does not exist, it will be created.
        log_options (dict): Additional keyword arguments to configure the logging.
            - "fast_logging": A boolean that determines if "unsafe" settings are used
            to speed up write processes to the database. This should only be used for
            very short running criterion functions where the main purpose of the log
            is monitoring and it would not be catastrophic to get a
            corrupted database in case of a sudden system shutdown. If one evaluation
            of the criterion function (and gradient if applicable) takes more than
            100 ms, the logging overhead is negligible.
            - "if_table_exists": (str) One of "extend", "replace", "raise". What to
            do if the tables we want to write to already exist. Default "extend".
            - "if_database_exists": (str): One of "extend", "replace", "raise". What to
            do if the database we want to write to already exists. Default "extend".
        loglike_kwargs (dict): Additional keyword arguments for loglike.
        jacobian (callable or None): A function that takes ``params`` and potentially
            other keyword arguments and returns the jacobian of loglike["contributions"]
            with respect to the params. Note that you only need to pass a Jacobian
            function if you have a closed form Jacobian. If you pass None, a numerical
            Jacobian will be calculated.
        jacobian_kwargs (dict): Additional keyword arguments for the Jacobian function.
        jacobian_numdiff_options (dict): Keyword arguments for the calculation of
            numerical derivatives for the calculation of standard errors. See
            :ref:`first_derivative` for details.
        hessian (callable or None or False): A function that takes ``params`` and
            potentially other keyword arguments and returns the Hessian of
            loglike["value"] with respect to the params.  If you pass None, a numerical
            Hessian will be calculated. If you pass ``False``, you signal that no
            Hessian should be calculated. Thus, no result that requires the Hessian will
            be calculated.
        hessian_kwargs (dict): Additional keyword arguments for the Hessian function.
        hessian_numdiff_options (dict): Keyword arguments for the calculation of
            numerical derivatives for the calculation of standard errors.
        design_info (pandas.DataFrame): DataFrame with one row per observation that
            contains some or all of the variables "psu" (primary sampling unit),
            "strata" and "fpc" (finite population corrector). See
            :ref:`robust_likelihood_inference` for details.

    Returns:
        LikelihoodResult: A LikelihoodResult object.

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
        deprecations.throw_numdiff_options_deprecated_in_estimate_ml_future_warning()
        if jacobian_numdiff_options is None:
            jacobian_numdiff_options = numdiff_options
        if hessian_numdiff_options is None:
            hessian_numdiff_options = numdiff_options

    deprecations.throw_dict_constraints_future_warning_if_required(constraints)

    # ==================================================================================
    # Check and process inputs
    # ==================================================================================

    loglike = mark.likelihood(loglike)

    bounds = pre_process_bounds(bounds)
    jacobian_numdiff_options = pre_process_numdiff_options(jacobian_numdiff_options)
    hessian_numdiff_options = pre_process_numdiff_options(hessian_numdiff_options)
    # TODO: Replace dict_constraints with constraints, once we deprecate dictionary
    # constraints.
    dict_constraints = deprecations.pre_process_constraints(constraints)

    if jacobian_numdiff_options is None:
        jacobian_numdiff_options = get_default_numdiff_options(
            purpose=NumdiffPurpose.ESTIMATE_JACOBIAN
        )

    if hessian_numdiff_options is None:
        hessian_numdiff_options = get_default_numdiff_options(
            purpose=NumdiffPurpose.ESTIMATE_HESSIAN
        )

    is_optimized = optimize_options is False

    if not is_optimized:
        # If optimize_options is not a dictionary and not False, we assume it represents
        # an algorithm. The actual testing of whether it is a valid algorithm is done
        # when `maximize` is called.
        if not isinstance(optimize_options, dict):
            optimize_options = {"algorithm": optimize_options}

        check_optimization_options(
            optimize_options,
            usage="estimate_ml",
            algorithm_mandatory=True,
        )

    jac_case = get_derivative_case(jacobian)
    hess_case = get_derivative_case(hessian)

    loglike_kwargs = {} if loglike_kwargs is None else loglike_kwargs
    jacobian_kwargs = {} if jacobian_kwargs is None else jacobian_kwargs
    hessian_kwargs = {} if hessian_kwargs is None else hessian_kwargs

    # ==================================================================================
    # Calculate estimates via maximization (if necessary)
    # ==================================================================================
    # Note: We do not need to handle deprecations for the optimization because that
    # is already done inside `maximize`.
    if is_optimized:
        estimates = params
        opt_res = None
    else:
        opt_res = maximize(
            fun=loglike,
            fun_kwargs=loglike_kwargs,
            params=params,
            bounds=bounds,
            constraints=constraints,
            logging=logging,
            log_options=log_options,
            **optimize_options,
        )
        estimates = opt_res.params

    # ==================================================================================
    # Do first function evaluations at estimated parameters
    # ==================================================================================

    try:
        loglike_eval = loglike(estimates, **loglike_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating loglike at estimated params."
        raise InvalidFunctionError(msg) from e

    if callable(jacobian):
        try:
            jacobian_eval = jacobian(estimates, **jacobian_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating closed form jacobian at estimated params."
            raise InvalidFunctionError(msg) from e
    else:
        jacobian_eval = None

    if callable(hessian):
        try:
            hessian_eval = hessian(estimates, **hessian_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating closed form hessian at estimated params."
            raise InvalidFunctionError(msg) from e
    else:
        hessian_eval = None

    # ==================================================================================
    # Handle deprecated function output
    # ==================================================================================
    if deprecations.is_dict_output(loglike_eval):
        deprecations.throw_dict_output_warning()
        loglike_eval = deprecations.convert_dict_to_function_value(loglike_eval)
        loglike = deprecations.replace_dict_output(loglike)
    else:
        loglike_eval = convert_fun_output_to_function_value(
            loglike_eval, AggregationLevel.LIKELIHOOD
        )
        loglike = enforce_return_type(AggregationLevel.LIKELIHOOD)(loglike)

    # ==================================================================================
    # Get the converter for params and function outputs
    # ==================================================================================

    converter, internal_estimates = get_converter(
        params=estimates,
        constraints=dict_constraints,
        bounds=bounds,
        func_eval=loglike_eval.value,
        solver_type="contributions",
        derivative_eval=jacobian_eval,
    )

    # ==================================================================================
    # Calculate internal jacobian
    # ==================================================================================

    if jac_case == "closed-form":
        int_jac = converter.derivative_to_internal(
            jacobian_eval, internal_estimates.values
        )
    elif jac_case == "numerical":

        def func(x):
            p = converter.params_from_internal(x)
            loglike_eval = loglike(p, **loglike_kwargs)
            if deprecations.is_dict_output(loglike_eval):
                deprecations.throw_dict_output_warning()
                loglike_eval = deprecations.convert_dict_to_function_value(loglike_eval)

            out = loglike_eval.internal_value(AggregationLevel.LIKELIHOOD)
            return out

        jac_res = first_derivative(
            func=func,
            params=internal_estimates.values,
            bounds=Bounds(
                lower=internal_estimates.lower_bounds,
                upper=internal_estimates.upper_bounds,
            ),
            error_handling="continue",
            **asdict(jacobian_numdiff_options),
        )

        int_jac = jac_res.derivative
    else:
        int_jac = None

    if dict_constraints in [None, []] and jacobian_eval is None and int_jac is not None:
        loglike_contribs = loglike_eval.value

        jacobian_eval = matrix_to_block_tree(
            int_jac,
            outer_tree=loglike_contribs,
            inner_tree=estimates,
        )

    if jacobian_eval is None:
        _no_jac_reason = (
            "no closed form jacobian was provided and there are constraints"
        )
    else:
        _no_jac_reason = None
    # ==================================================================================
    # Calculate internal Hessian
    # ==================================================================================

    if hess_case == "skip":
        int_hess = None
    elif hess_case == "numerical":

        def func(x):
            p = converter.params_from_internal(x)
            loglike_eval = loglike(p, **loglike_kwargs)
            if deprecations.is_dict_output(loglike_eval):
                deprecations.throw_dict_output_warning()
                loglike_eval = deprecations.convert_dict_to_function_value(loglike_eval)

            out = loglike_eval.internal_value(AggregationLevel.SCALAR)
            return out

        hess_res = second_derivative(
            func=func,
            params=internal_estimates.values,
            bounds=Bounds(
                lower=internal_estimates.lower_bounds,
                upper=internal_estimates.upper_bounds,
            ),
            error_handling="continue",
            **asdict(hessian_numdiff_options),
        )
        int_hess = hess_res.derivative
    elif hess_case == "closed-form" and dict_constraints:
        raise NotImplementedError(
            "Closed-form Hessians are not yet compatible with constraints."
        )
    elif hess_case == "closed-form":
        int_hess = block_tree_to_matrix(
            hessian_eval,
            outer_tree=params,
            inner_tree=params,
        )
    else:
        raise ValueError()

    if dict_constraints in [None, []] and hessian_eval is None and int_hess is not None:
        hessian_eval = matrix_to_block_tree(
            int_hess,
            outer_tree=params,
            inner_tree=params,
        )

    if hessian_eval is None:
        if hess_case == "skip":
            _no_hess_reason = "the hessian calculation was explicitly skipped."
        else:
            _no_hess_reason = (
                "no closed form hessian was provided and there are constraints"
            )
    else:
        _no_hess_reason = None

    # ==================================================================================
    # create a LikelihoodResult object
    # ==================================================================================

    free_estimates = calculate_free_estimates(estimates, internal_estimates)

    res = LikelihoodResult(
        _params=estimates,
        _converter=converter,
        _optimize_result=opt_res,
        _jacobian=jacobian_eval,
        _no_jacobian_reason=_no_jac_reason,
        _hessian=hessian_eval,
        _no_hessian_reason=_no_hess_reason,
        _internal_jacobian=int_jac,
        _internal_hessian=int_hess,
        _design_info=design_info,
        _internal_estimates=internal_estimates,
        _free_estimates=free_estimates,
        _has_constraints=dict_constraints not in [None, []],
    )

    return res


@dataclass
class LikelihoodResult:
    """Likelihood estimation results object."""

    _params: Any
    _internal_estimates: InternalParams
    _free_estimates: FreeParams
    _converter: Converter
    _has_constraints: bool
    _optimize_result: OptimizeResult | None = None
    _jacobian: Any = None
    _no_jacobian_reason: str | None = None
    _hessian: Any = None
    _no_hessian_reason: str | None = None
    _internal_jacobian: np.ndarray | None = None
    _internal_hessian: np.ndarray | None = None
    _design_info: pd.DataFrame | None = None
    _cache: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self._internal_jacobian is None and self._internal_hessian is None:
            raise ValueError(
                "At least one of _internal_jacobian or _internal_hessian must be "
                "not None."
            )

        elif self._internal_jacobian is None:
            valid_methods = ["hessian"]
        elif self._internal_hessian is None:
            valid_methods = ["jacobian"]
        else:
            valid_methods = ["jacobian", "hessian", "robust"]
            if self._design_info is not None:
                if "psu" in self._design_info:
                    valid_methods.append("cluster_robust")
                if {"strata", "psu", "fpc"}.issubset(self._design_info):
                    valid_methods.append("strata_robust")

        self._valid_methods = set(valid_methods)

    def _get_free_cov(
        self,
        method,
        n_samples,
        bounds_handling,
        seed,
    ):
        if method not in self._valid_methods:
            msg = f"Invalid method: {method}. Valid methods are {self._valid_methods}."
            raise ValueError(msg)
        args = (method, n_samples, bounds_handling, seed)
        is_cached = args in self._cache

        if is_cached:
            free_cov = self._cache[args]
        else:
            free_cov = _calculate_free_cov_ml(
                method=method,
                internal_estimates=self._internal_estimates,
                converter=self._converter,
                internal_jacobian=self._internal_jacobian,
                internal_hessian=self._internal_hessian,
                n_samples=n_samples,
                design_info=self._design_info,
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
    def jacobian(self):
        if self._jacobian is None:
            raise NotAvailableError(
                f"No jacobian is available because {self._no_jacobian_reason}."
            )
        return self._jacobian

    @property
    def hessian(self):
        if self._hessian is None:
            raise NotAvailableError(
                f"No hessian is available because {self._no_hessian_reason}."
            )
        return self._hessian

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
        method="jacobian",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate standard errors.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
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
        method="jacobian",
        n_samples=10_000,
        bounds_handling="clip",
        return_type="pytree",
        seed=None,
    ):
        """Calculate the variance-covariance (matrix) of the estimated parameters.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
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
            Any: The covariance matrix of the estimated parameters as block-pytree,
                numpy.ndarray or pandas.DataFrame.

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
        method="jacobian",
        n_samples=10_000,
        ci_level=0.95,
        bounds_handling="clip",
        seed=None,
    ):
        """Create a summary of estimation results.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
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
        method="jacobian",
        n_samples=10_000,
        ci_level=0.95,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate confidence intervals.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
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
        method="jacobian",
        n_samples=10_000,
        bounds_handling="clip",
        seed=None,
    ):
        """Calculate p-values.

        Args:
            method (str): One of "jacobian", "hessian", "robust", "cluster_robust",
                "strata_robust". Default "jacobian". "cluster_robust" is only available
                if design_info containts a columns called "psu" that identifies the
                primary sampling unit. "strata_robust" is only available if the columns
                "strata", "fpc" and "psu" are in design_info.
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

    def to_pickle(self, path):
        """Save the LikelihoodResult object to pickle.

        Args:
            path (str, pathlib.Path): A str or pathlib.path ending in .pkl or .pickle.

        """
        to_pickle(self, path=path)


def _calculate_free_cov_ml(
    method,
    internal_estimates,
    converter,
    internal_jacobian,
    internal_hessian,
    n_samples,
    design_info,
    bounds_handling,
    seed,
):
    if method == "jacobian":
        int_cov = cov_jacobian(internal_jacobian)
    elif method == "hessian":
        int_cov = cov_hessian(internal_hessian)
    elif method == "robust":
        int_cov = cov_robust(jac=internal_jacobian, hess=internal_hessian)
    elif method == "cluster_robust":
        int_cov = cov_cluster_robust(
            jac=internal_jacobian, hess=internal_hessian, design_info=design_info
        )
    elif method == "strata_robust":
        int_cov = cov_strata_robust(
            jac=internal_jacobian, hess=internal_hessian, design_info=design_info
        )

    rng = get_rng(seed)

    free_cov = transform_covariance(
        internal_params=internal_estimates,
        internal_cov=int_cov,
        converter=converter,
        rng=rng,
        n_samples=n_samples,
        bounds_handling=bounds_handling,
    )
    return free_cov
