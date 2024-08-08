import functools

import numpy as np
import pandas as pd
from pybaum import tree_just_flatten
from scipy.linalg import block_diag

from estimagic.inference.bootstrap import bootstrap
from estimagic.parameters.block_trees import block_tree_to_matrix, matrix_to_block_tree
from estimagic.parameters.tree_registry import get_registry
from estimagic.utilities import robust_inverse


def get_moments_cov(
    data, calculate_moments, *, moment_kwargs=None, bootstrap_kwargs=None
):
    """Bootstrap the covariance matrix of the moment conditions.

    Args:
        data (pandas.DataFrame): DataFrame with empirical data.
        calculate_moments (callable): Function that calculates that takes data and
            moment_kwargs as arguments and returns a 1d numpy array or pandas Series
            with moment conditions.
        moment_kwargs (dict): Additional keyword arguments for calculate_moments.
        bootstrap_kwargs (dict): Additional keyword arguments that govern the
            bootstrapping. Allowed arguments are "n_draws", "seed", "n_cores",
            "batch_evaluator", "weight_by", "cluster_by" and "error_handling".
            For details see the bootstrap function.

    Returns:
        pandas.DataFrame or numpy.ndarray: The covariance matrix of the moment
            conditions for msm estimation.

    """
    moment_kwargs = {} if moment_kwargs is None else moment_kwargs
    bootstrap_kwargs = {} if bootstrap_kwargs is None else bootstrap_kwargs
    valid_bs_kwargs = {
        "n_cores",
        "n_draws",
        "seed",
        "batch_evaluator",
        "weight_by",
        "cluster_by",
        "error_handling",
        "existing_result",
        "outcome_kwargs",
    }
    problematic = set(bootstrap_kwargs).difference(valid_bs_kwargs)
    if problematic:
        raise ValueError(f"Invalid bootstrap_kwargs: {problematic}")

    first_eval = calculate_moments(data, **moment_kwargs)

    registry = get_registry(extended=True)

    @functools.wraps(calculate_moments)
    def func(data, **kwargs):
        raw = calculate_moments(data, **kwargs)
        out = pd.Series(
            tree_just_flatten(raw, registry=registry)
        )  # xxxx won't be necessary soon!
        return out

    cov_arr = bootstrap(
        data=data, outcome=func, outcome_kwargs=moment_kwargs, **bootstrap_kwargs
    ).cov()

    if isinstance(cov_arr, pd.DataFrame):
        cov_arr = cov_arr.to_numpy()  # xxxx won't be necessary soon

    cov = matrix_to_block_tree(cov_arr, first_eval, first_eval)

    return cov


def get_weighting_matrix(
    moments_cov, method, empirical_moments, clip_value=1e-6, return_type="pytree"
):
    """Calculate a weighting matrix from moments_cov.

    Args:
        moments_cov (pandas.DataFrame or numpy.ndarray): Square DataFrame or Array
            with the covariance matrix of the moment conditions for msm estimation.
        method (str): One of "optimal", "diagonal", or "identity".
        empirical_moments (pytree): Pytree containing empirical moments. Used to get
            the tree structure
        clip_value (float): Bound at which diagonal elements of the moments_cov are
            clipped to avoid dividing by zero.
        return_type (str): One of "pytree", "array" or "pytree_and_array"

    Returns:
        pandas.DataFrame or numpy.ndarray: Weighting matrix with the same shape as
            moments_cov.

    """
    fast_path = isinstance(moments_cov, np.ndarray) and moments_cov.ndim == 2

    if fast_path:
        _internal_cov = moments_cov
    else:
        _internal_cov = block_tree_to_matrix(
            moments_cov,
            outer_tree=empirical_moments,
            inner_tree=empirical_moments,
        )

    if method == "optimal":
        array_weights = robust_inverse(_internal_cov)
    elif method == "diagonal":
        diagonal_values = 1 / np.clip(np.diagonal(_internal_cov), clip_value, np.inf)
        array_weights = np.diag(diagonal_values)
    elif method == "identity":
        array_weights = np.identity(_internal_cov.shape[0])
    else:
        raise ValueError(f"Invalid method: {method}")

    if return_type == "array" or (fast_path and "_and_" not in return_type):
        out = array_weights
    elif fast_path:
        out = (array_weights, array_weights)
    else:
        tree_weights = matrix_to_block_tree(
            array_weights,
            outer_tree=empirical_moments,
            inner_tree=empirical_moments,
        )
        if return_type == "pytree":
            out = tree_weights
        else:
            out = (tree_weights, array_weights)

    return out


def _assemble_block_diagonal_matrix(matrices):
    """Build a block diagonal matrix out of matrices.

    Args:
        matrices (list): List of square numpy arrays or DataFrames with the building
            blocks for the block diagonal matrix.

    Returns:
        pandas.DataFrame or numpy.ndarray: The block diagonal matrix.

    """
    values = block_diag(*matrices)

    if all(isinstance(mat, pd.DataFrame) for mat in matrices):
        to_concat = [pd.Series(index=mat.index, dtype=float) for mat in matrices]
        combined_index = pd.concat(to_concat).index
        out = pd.DataFrame(values, index=combined_index, columns=combined_index)
    else:
        out = values
    return out
