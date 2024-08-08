from estimagic.batch_evaluators import process_batch_evaluator
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_samples import get_bootstrap_indices


def get_bootstrap_outcomes(
    data,
    outcome,
    weight_by=None,
    cluster_by=None,
    rng=None,
    n_draws=1000,
    n_cores=1,
    error_handling="continue",
    batch_evaluator="joblib",
):
    """Draw bootstrap samples and calculate outcomes.

    Args:
        data (pandas.DataFrame): original dataset.
        outcome (callable): function of the dataset calculating statistic of interest.
            Returns a general pytree (e.g. pandas Series, dict, numpy array, etc.).
        weight_by (str): column name of the variable with weights.
        cluster_by (str): column name of the variable to cluster by.
        rng (numpy.random.Generator): A random number generator.
        n_draws (int): number of bootstrap draws.
        n_cores (int): number of jobs for parallelization.
        error_handling (str): One of "continue", "raise". Default "continue" which means
            that bootstrap estimates are only calculated for those samples where no
            errors occur and a warning is produced if any error occurs.
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.

    Returns:
        estimates (list):  List of pytrees of estimated bootstrap outcomes.

    """
    check_inputs(data=data, weight_by=weight_by, cluster_by=cluster_by)
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    indices = get_bootstrap_indices(
        data=data,
        rng=rng,
        weight_by=weight_by,
        cluster_by=cluster_by,
        n_draws=n_draws,
    )

    estimates = _get_bootstrap_outcomes_from_indices(
        indices=indices,
        data=data,
        outcome=outcome,
        n_cores=n_cores,
        error_handling=error_handling,
        batch_evaluator=batch_evaluator,
    )

    return estimates


def _get_bootstrap_outcomes_from_indices(
    indices,
    data,
    outcome,
    n_cores,
    error_handling,
    batch_evaluator,
):
    arguments = [{"data": data, "indices": ind, "outcome": outcome} for ind in indices]

    raw_estimates = batch_evaluator(
        _take_indices_and_calculate_outcome,
        arguments,
        n_cores=n_cores,
        unpack_symbol="**",
        error_handling=error_handling,
    )

    estimates = [est for est in raw_estimates if not isinstance(est, str)]
    tracebacks = [est for est in raw_estimates if isinstance(est, str)]

    if not estimates:
        msg = (
            "Calculating of all bootstrap outcomes failed. The tracebacks of the "
            "raised Exceptions are reproduced below:"
        )
        raise RuntimeError(msg + "\n\n" + "\n\n".join(tracebacks))

    if tracebacks:
        msg = (
            "Calculating bootstrap outcomes failed for some samples. Those samples "
            "are excluded from the calculation of bootstrap standard errors and "
            "confidence intervals, rendering them invalid. Do not use them for "
            "anything but diagnostic purposes. Check warnings for more information. "
        )

    return estimates


def _take_indices_and_calculate_outcome(indices, data, outcome):
    return outcome(data.iloc[indices])
