import functools
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from pybaum import leaf_names, tree_flatten, tree_just_flatten, tree_unflatten

from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.inference.bootstrap_ci import calculate_ci
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_outcomes import get_bootstrap_outcomes
from estimagic.inference.shared import calculate_estimation_summary
from estimagic.parameters.block_trees import matrix_to_block_tree
from estimagic.parameters.tree_registry import get_registry
from estimagic.utilities import get_rng


def bootstrap(
    outcome,
    data,
    *,
    existing_result=None,
    outcome_kwargs=None,
    n_draws=1_000,
    weight_by=None,
    cluster_by=None,
    seed=None,
    n_cores=1,
    error_handling="continue",
    batch_evaluator=joblib_batch_evaluator,
):
    """Use the bootstrap to calculate inference quantities.

    Args:
        outcome (callable): A function that computes the statistic of interest.
        data (pd.DataFrame): Dataset.
        existing_result (BootstrapResult): An existing BootstrapResult
            object from a previous call of bootstrap(). Default is None.
        outcome_kwargs (dict): Additional keyword arguments for outcome.
        n_draws (int): Number of bootstrap samples to draw.
            If len(existing_outcomes) >= n_draws, a random subset of existing_outcomes
            is used.
        weight_by (str): Column name of variable with weights or None.
        cluster_by (str): Column name of variable to cluster by or None.
        seed (Union[None, int, numpy.random.Generator]): If seed is None or int the
            numpy.random.default_rng is used seeded with seed. If seed is already a
            Generator instance then that instance is used.
        n_cores (int): number of jobs for parallelization.
        error_handling (str): One of "continue", "raise". Default "continue" which means
            that bootstrap estimates are only calculated for those samples where no
            errors occur and a warning is produced if any error occurs.
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.

    Returns:
        BootstrapResult: A BootstrapResult object storing information on summary
            statistics, the covariance matrix, and estimated boostrap outcomes.

    """
    if callable(outcome):
        check_inputs(data=data, weight_by=weight_by, cluster_by=cluster_by)

        if outcome_kwargs is not None:
            outcome = functools.partial(outcome, **outcome_kwargs)
    else:
        raise TypeError("outcome must be a callable.")

    if existing_result is None:
        base_outcome = outcome(data)
        existing_outcomes = []
    elif isinstance(existing_result, BootstrapResult):
        base_outcome = existing_result.base_outcome
        existing_outcomes = existing_result.outcomes
    else:
        raise ValueError("existing_result must be None or a BootstrapResult.")

    rng = get_rng(seed)
    n_existing = len(existing_outcomes)

    if n_draws > n_existing:
        new_outcomes = get_bootstrap_outcomes(
            data=data,
            outcome=outcome,
            weight_by=weight_by,
            cluster_by=cluster_by,
            rng=rng,
            n_draws=n_draws - n_existing,
            n_cores=n_cores,
            error_handling=error_handling,
            batch_evaluator=batch_evaluator,
        )

        all_outcomes = existing_outcomes + new_outcomes
    else:
        random_indices = rng.choice(n_existing, n_draws, replace=False)
        all_outcomes = [existing_outcomes[k] for k in random_indices]

    # ==================================================================================
    # Process results
    # ==================================================================================

    registry = get_registry(extended=True)
    flat_outcomes = [
        tree_just_flatten(_outcome, registry=registry) for _outcome in all_outcomes
    ]
    internal_outcomes = np.array(flat_outcomes)

    result = BootstrapResult(
        _base_outcome=base_outcome,
        _internal_outcomes=internal_outcomes,
        _internal_cov=np.cov(internal_outcomes, rowvar=False),
    )

    return result


@dataclass
class BootstrapResult:
    _base_outcome: Any
    _internal_outcomes: np.ndarray
    _internal_cov: np.ndarray

    @cached_property
    def _se(self):
        return self.se()

    @cached_property
    def _cov(self):
        return self.cov()

    @cached_property
    def _ci(self):
        return self.ci()

    @cached_property
    def _p_values(self):
        return self.p_values()

    @cached_property
    def _summary(self):
        return self.summary()

    @property
    def base_outcome(self):
        """Returns the base outcome statistic(s).

        Returns:
            pytree: Pytree of base outcomes, i.e. the outcome statistic(s) evaluated
                on the original data set.

        """
        return self._base_outcome

    @cached_property
    def outcomes(self):
        """Returns the estimated bootstrap outcomes.

        Returns:
            List[Any]: The boostrap outcomes as a list of pytrees.

        """
        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self._base_outcome, registry=registry)

        outcomes = [
            tree_unflatten(treedef, out, registry=registry)
            for out in self._internal_outcomes
        ]
        return outcomes

    def se(self):
        """Calculate standard errors.

        Returns:
            Any: The standard errors of the estimated parameters as a block-pytree,
                numpy.ndarray, or pandas.DataFrame.

        """
        cov = self._internal_cov
        se = np.sqrt(np.diagonal(cov))

        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self._base_outcome, registry=registry)

        se = tree_unflatten(treedef, se, registry=registry)
        return se

    def cov(self, return_type="pytree"):
        """Calculate the variance-covariance matrix of the estimated parameters.

        Args:
            return_type (str): One of "pytree", "array" or "dataframe". Default pytree.
                If "array", a 2d numpy array with the covariance is returned. If
                "dataframe", a pandas DataFrame with parameter names in the
                index and columns are returned.
                The default is "pytree".

        Returns:
            Any: The covariance matrix of the estimated parameters as a block-pytree,
                numpy.ndarray, or pandas.DataFrame.

        """
        cov = self._internal_cov

        if return_type == "dataframe":
            registry = get_registry(extended=True)
            names = np.array(leaf_names(self._base_outcome, registry=registry))
            cov = pd.DataFrame(cov, columns=names, index=names)
        elif return_type == "pytree":
            cov = matrix_to_block_tree(cov, self._base_outcome, self._base_outcome)
        elif return_type != "array":
            raise ValueError(
                "return_type must be one of pytree, array, or dataframe, "
                f"not {return_type}."
            )
        return cov

    def ci(self, ci_method="percentile", ci_level=0.95):
        """Calculate confidence intervals.

        Args:
            ci_method (str): Method of choice for computing confidence intervals.
                The default is "percentile".
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.

        Returns:
            Any: Pytree with the same structure as base_outcome containing lower
                bounds of confidence intervals.
            Any: Pytree with the same structure as base_outcome containing upper
                bounds of confidence intervals.

        """
        registry = get_registry(extended=True)
        base_outcome_flat, treedef = tree_flatten(self._base_outcome, registry=registry)

        lower_flat, upper_flat = calculate_ci(
            base_outcome_flat, self._internal_outcomes, ci_method, ci_level
        )

        lower = tree_unflatten(treedef, lower_flat, registry=registry)
        upper = tree_unflatten(treedef, upper_flat, registry=registry)
        return lower, upper

    def p_values(self):
        """Calculate p-values.

        Returns:
            Any: A pytree with the same structure as base_outcome containing p-values
                for the parameter estimates.

        """
        msg = "Bootstrap p_values are not yet implemented."
        raise NotImplementedError(msg)

    def summary(self, ci_method="percentile", ci_level=0.95):
        """Create a summary of bootstrap results.

        Args:
            ci_method (str): Method of choice for confidence interval computation.
                The default is "percentile".
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.

        Returns:
            pd.DataFrame: The estimation summary as a DataFrame containing information
                on the mean, standard errors, as well as the confidence intervals.
                Soon this will be a pytree.

        """
        registry = get_registry(extended=True)
        names = leaf_names(self.base_outcome, registry=registry)
        summary_data = _calulcate_summary_data_bootstrap(
            self, ci_method=ci_method, ci_level=ci_level
        )
        summary = calculate_estimation_summary(
            summary_data=summary_data,
            names=names,
            free_names=names,
        )
        return summary


def _calulcate_summary_data_bootstrap(bootstrap_result, ci_method, ci_level):
    lower, upper = bootstrap_result.ci(ci_method=ci_method, ci_level=ci_level)
    summary_data = {
        "value": bootstrap_result.base_outcome,
        "standard_error": bootstrap_result.se(),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": np.full(len(lower), np.nan),  # p-values are not implemented yet
    }
    return summary_data
