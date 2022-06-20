from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy
from estimagic.parameters.block_trees import matrix_to_block_tree
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten
from pybaum import tree_unflatten


def transform_covariance(
    internal_params,
    internal_cov,
    converter,
    n_samples,
    bounds_handling,
):
    """Transform the internal covariance matrix to an external one, given constraints.

    Args:
        internal_params (InternalParams): NamedTuple with entries:
            - value (np.ndarray): Internal parameter values.
            - lower_bounds (np.ndarray): Lower bounds on the internal params.
            - upper_bounds (np.ndarray): Upper bounds on the internal params.
            - soft_lower_bounds (np.ndarray): Soft lower bounds on the internal params.
            - soft_upper_bounds (np.ndarray): Soft upper bounds on the internal params.
            - name (list): List of names of the external parameters.
            - free_mask (np.ndarray): Boolean mask representing which external parameter
              is free.
        internal_cov (np.ndarray or pandas.DataFrame) with a covariance matrix of the
            internal parameter vector. For background information about internal and
            external params see :ref:`implementation_of_constraints`.
        constraints (list): List with constraint dictionaries.
            See :ref:`constraints`.
        n_samples (int): Number of samples used to transform the covariance matrix of
            the internal parameter vector into the covariance matrix of the external
            parameters.
        bounds_handling (str): One of "clip", "raise", "ignore". Determines how bounds
            are handled. If "clip", confidence intervals are clipped at the bounds.
            Standard errors are only adjusted if a sampling step is necessary due to
            additional constraints. If "raise" and any lower or upper bound is binding,
            we raise an error. If "ignore", boundary problems are simply ignored.

    Returns:
        pd.DataFrame: Quadratic DataFrame containing the covariance matrix of the free
            parameters. If parameters were fixed (explicitly or by other constraints),
            the index is a subset of params.index. The columns are the same as the
            index.
    """
    if isinstance(internal_cov, pd.DataFrame):
        internal_cov = internal_cov.to_numpy()

    if converter.has_transforming_constraints:
        _from_internal = converter.params_from_internal

        is_free = internal_params.free_mask
        lower_bounds = internal_params.lower_bounds
        upper_bounds = internal_params.upper_bounds

        sample = np.random.multivariate_normal(
            mean=internal_params.values,
            cov=internal_cov,
            size=n_samples,
        )
        transformed_free = []
        for params_vec in sample:
            if bounds_handling == "clip":
                params_vec = np.clip(params_vec, a_min=lower_bounds, a_max=upper_bounds)
            elif bounds_handling == "raise":
                if (params_vec < lower_bounds).any() or (
                    params_vec > upper_bounds
                ).any():
                    raise ValueError()

            transformed = _from_internal(x=params_vec, return_type="flat")
            transformed_free.append(transformed[is_free])

        free_cov = np.cov(
            np.array(transformed_free),
            rowvar=False,
        )

    else:
        free_cov = internal_cov

    return free_cov


def calculate_estimation_summary(estimates, internal_estimates, free_cov, ci_level):
    """Add standard errors, pvalues and confidence intervals to params.

    Args
        estimates (pytree): The input parameter pytree.
        internal_estimates (FlatParams): NamedTuple with internal estimated parameter
            values and names, lower_bounds and upper_bounds, and free_mask.
        free_cov (pd.DataFrame): Quadratic DataFrame containing the covariance matrix
            of the free parameters. If parameters were fixed (explicitly or by other
            constraints) the index is a subset of params.index. The columns are the same
            as the index.
        ci_level (float): Confidence level for the calculation of confidence intervals.

    Returns:
        pytree: A pytree with the same structure as params. Each leaf in the params
            tree is replaced by a DataFrame containing columns "value",
            "standard_error", "pvalue", "ci_lower" and "ci_upper".  Parameters that do
            not have a standard error (e.g. because they were fixed during estimation)
            contain NaNs in all but the "value" column. The value column is only
            reproduced for convenience.

    """
    if not isinstance(free_cov, pd.DataFrame):
        free_index = np.array(internal_estimates.free_names)
        free_cov = pd.DataFrame(data=free_cov, columns=free_index, index=free_index)

    # ==================================================================================
    # Construct summary data frame for flat estimates
    # ==================================================================================
    registry = get_registry(extended=True)

    df = pd.DataFrame(index=internal_estimates.allnames)
    df["value"] = tree_just_flatten(estimates, registry=registry)
    df["free"] = internal_estimates.free_mask
    df.loc[free_cov.index, "standard_error"] = np.sqrt(np.diag(free_cov))

    df["p_value"] = calculate_p_values(
        df["value"].to_numpy(),
        df["standard_error"].to_numpy(),
    )

    lower, upper = calculate_ci(
        df["value"].to_numpy(),
        df["standard_error"].to_numpy(),
        ci_level=ci_level,
    )
    df["ci_lower"] = lower
    df["ci_upper"] = upper

    df.loc[free_cov.index, "stars"] = pd.cut(
        df.loc[free_cov.index, "p_value"],
        bins=[-1, 0.01, 0.05, 0.1, 2],
        labels=["***", "**", "*", ""],
    )

    # ==================================================================================
    # Map summary data into params tree structure
    # ==================================================================================

    # create tree with values corresponding to indices of df
    indices = tree_unflatten(estimates, internal_estimates.names, registry=registry)

    estimates_flat = tree_just_flatten(estimates)
    indices_flat = tree_just_flatten(indices)

    # use index chunks in indices_flat to access the corresponding sub data frame of df,
    # and use the index information stored in estimates_flat to form the correct (multi)
    # index for the resulting leaf.
    summary_flat = []
    for index_leaf, params_leaf in zip(indices_flat, estimates_flat):

        if np.isscalar(params_leaf):
            loc = [index_leaf]
            index = [0]
        elif isinstance(params_leaf, pd.DataFrame) and "value" in params_leaf:
            loc = index_leaf["value"].values.flatten()
            index = params_leaf.index
        elif isinstance(params_leaf, pd.DataFrame):
            loc = index_leaf.values.flatten()
            # use product of existing index and columns for regular pd.DataFrame
            index = pd.MultiIndex.from_tuples(
                [
                    (*row, col)
                    for row in params_leaf.index
                    for col in params_leaf.columns
                ]
            )
        elif isinstance(params_leaf, pd.Series):
            loc = index_leaf.values.flatten()
            index = params_leaf.index
        else:
            # array case (numpy or jax)
            loc = index_leaf.flatten()
            if params_leaf.ndim == 1:
                index = pd.RangeIndex(stop=params_leaf.size)
            else:
                index = pd.MultiIndex.from_arrays(
                    np.unravel_index(np.arange(params_leaf.size), params_leaf.shape)
                )

        df_chunk = df.loc[loc]
        df_chunk.index = index

        summary_flat.append(df_chunk)

    summary = tree_unflatten(estimates, summary_flat)
    return summary


def process_pandas_arguments(**kwargs):
    """Convert pandas objects to arrays and extract names of moments and parameters.

    This works for any number of keyword arguments. The result is a tuple containing
    numpy arrays in same order as the keyword arguments and a dictionary with
    the separated index objects as last entry. This dictionary contains the entries
    "moments" and "params" for the identified moment names and parameter names.

    The keyword arguments "jac", "hess", "weights" and "moments_cov" are used to extract
    the names. Other keyword arguments are simply converted to numpy arrays.

    """
    param_name_candidates = {}
    moment_name_candidates = {}

    if "jac" in kwargs:
        jac = kwargs["jac"]
        if isinstance(jac, pd.DataFrame):
            param_name_candidates["jac"] = jac.columns
            moment_name_candidates["jac"] = jac.index

    if "hess" in kwargs:
        hess = kwargs["hess"]
        if isinstance(hess, pd.DataFrame):
            param_name_candidates["hess"] = hess.index

    if "weights" in kwargs:
        weights = kwargs["weights"]
        if isinstance(weights, pd.DataFrame):
            moment_name_candidates["weights"] = weights.index

    if "moments_cov" in kwargs:
        moments_cov = kwargs["moments_cov"]
        if isinstance(moments_cov, pd.DataFrame):
            moment_name_candidates["moments_cov"] = moments_cov.index

    names = {}
    if param_name_candidates:
        _check_names_coincide(param_name_candidates)
        names["params"] = list(param_name_candidates.values())[0]
    if moment_name_candidates:
        _check_names_coincide(moment_name_candidates)
        names["moments"] = list(moment_name_candidates.values())[0]

    # order of outputs is same as order of inputs; names are last.
    out_list = [_to_numpy(val, name=key) for key, val in kwargs.items()] + [names]
    return tuple(out_list)


def _to_numpy(df_or_array, name):
    if isinstance(df_or_array, pd.DataFrame):
        arr = df_or_array.to_numpy()
    elif isinstance(df_or_array, np.ndarray):
        arr = df_or_array
    else:
        raise ValueError(
            f"{name} must be a DataFrame or numpy array, not {type(df_or_array)}."
        )
    return arr


def _check_names_coincide(name_dict):
    if len(name_dict) >= 2:
        first_key = list(name_dict)[0]
        first_names = name_dict[first_key]

        for key, names in name_dict.items():
            if not first_names.equals(names):
                msg = f"Ambiguous parameter or moment names from {first_key} and {key}."
                raise ValueError(msg)


def get_derivative_case(derivative):
    """Determine which kind of derivative should be used."""
    if isinstance(derivative, (pd.DataFrame, np.ndarray)):
        case = "pre-calculated"
    elif callable(derivative):
        case = "closed-form"
    elif derivative is False:
        case = "skip"
    else:
        case = "numerical"
    return case


def check_is_optimized_and_derivative_case(is_minimized, derivative_case):
    if (not is_minimized) and derivative_case == "pre-calculated":
        raise ValueError(
            "Providing a pre-calculated derivative is only possible if the "
            "optimization was done outside of the estimate_function, i.e. if "
            "optimize_options=False."
        )


def calculate_ci(flat_values, flat_standard_errors, ci_level):
    alpha = 1 - ci_level
    scale = scipy.stats.norm.ppf(1 - alpha / 2)
    lower = flat_values - scale * flat_standard_errors
    upper = flat_values + scale * flat_standard_errors
    return lower, upper


def calculate_p_values(flat_values, flat_standard_error):
    tvalues = flat_values / np.clip(flat_standard_error, 1e-300, np.inf)
    pvalues = 2 * scipy.stats.norm.sf(np.abs(tvalues))
    return pvalues


def calculate_free_estimates(estimates, internal_estimates):
    mask = internal_estimates.free_mask
    names = internal_estimates.names

    registry = get_registry(extended=True)
    external_flat = np.array(tree_just_flatten(estimates, registry=registry))

    free_estimates = FreeParams(
        values=external_flat[mask],
        free_mask=mask,
        all_names=names,
        free_names=np.array(names)[mask].tolist(),
    )
    return free_estimates


def transform_free_cov_to_cov(free_cov, free_params, params, return_type):
    mask = free_params.free_mask
    cov = np.full((len(mask), len(mask)), np.nan)
    cov[np.ix_(mask, mask)] = free_cov
    if return_type == "dataframe":
        names = free_params.all_names
        cov = pd.DataFrame(cov, columns=names, index=names)
    elif return_type == "pytree":
        cov = matrix_to_block_tree(cov, params, params)
    return cov


def transform_free_params_to_params(free_params, params):
    mask = free_params.free_mask
    flat = np.full(len(mask), np.nan)
    flat[np.ix_(mask)] = free_params.values
    registry = get_registry(extended=True)
    pytree = tree_unflatten(params, flat, registry=registry)
    return pytree


class FreeParams(NamedTuple):
    values: np.ndarray  # free external parameter values
    free_mask: np.ndarray  # boolean mask to filter free params from external params
    free_names: list  # names of free external parameters
    all_names: list  # names of all external parameters
