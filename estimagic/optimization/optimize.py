"""Functional wrapper around the pygmo, nlopt and scipy libraries."""
import json
from collections import namedtuple
from multiprocessing import Event
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd

from estimagic.dashboard.server_functions import run_server
from estimagic.optimization.pounders import minimize_pounders
from estimagic.optimization.process_arguments import process_optimization_arguments
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.pygmo import minimize_pygmo
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal
from estimagic.optimization.scipy import minimize_scipy
from estimagic.optimization.utilities import index_element_to_string
from estimagic.optimization.utilities import propose_algorithms

QueueEntry = namedtuple("QueueEntry", ["iteration", "params", "fitness"])


def maximize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=False,
    db_options=None,
):
    """
    Maximize *criterion* using *algorithm* subject to *pc* and bounds.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        constraints (list):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization.

        algo_options (dict):
            algorithm specific configurations for the optimization.

        dashboard (bool):
            whether to create and show a dashboard.

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """

    def neg_criterion(params, **criterion_kwargs):
        return -criterion(params, **criterion_kwargs)

    # identify the criterion function as belongig to a maximization problem
    if general_options is None:
        general_options = {"_maximization": True}
    else:
        general_options["_maximization"] = True

    results, params = minimize(
        neg_criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        dashboard=dashboard,
        db_options=db_options,
    )
    results["fitness"] = -results["fitness"]

    return results, params


def minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=False,
    db_options=None,
):
    """Minimize *criterion* using *algorithm* subject to *constraints* and bounds.
    Run several optimizations if called by lists of inputs.

    Args:
        criterion (function or list of functions):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.

        algorithm (str or list of strings):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_kwargs (dict or list of dicts):
            additional keyword arguments for criterion

        constraints (list or list of lists):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict or list of dicts):
            algorithm specific configurations for the optimization

        dashboard (bool):
            whether to create and show a dashboard

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """

    arguments = process_optimization_arguments(
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        dashboard=dashboard,
        db_options=db_options,
    )

    if len(arguments) == 1:
        # Run only one optimization
        arguments = arguments[0]
        result = _single_minimize(**arguments)
    else:
        # Run multiple optimizations
        if dashboard:
            raise NotImplementedError(
                "Dashboard cannot be used for multiple optimizations, yet."
            )

        # set up multiprocessing
        if "n_cores" not in arguments[0]["general_options"]:
            raise ValueError(
                "n_cores need to be specified in general_options"
                + " if multiple optimizations should be run."
            )
        n_cores = arguments[0]["general_options"]["n_cores"]
        pool = Pool(processes=n_cores)
        result = pool.map(_one_argument_single_minimize, arguments)

    return result


def _single_minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs,
    constraints,
    general_options,
    algo_options,
    dashboard,
    db_options,
):
    """Minimize * criterion * using * algorithm * subject to * constraints * and bounds.
    Only one minimization.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        constraints (list):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict):
            algorithm specific configurations for the optimization

        dashboard (bool):
            whether to create and show a dashboard

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.

    """
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    params = _process_params(params)

    fitness_factor = -1 if general_options.get("_maximization", False) else 1
    fitness_eval = fitness_factor * criterion(params, **criterion_kwargs)
    constraints, params = process_constraints(constraints, params)
    internal_params = reparametrize_to_internal(params, constraints)

    queue = Queue() if dashboard else None
    if dashboard:
        stop_signal = Event()
        outer_server_process = Process(
            target=run_server,
            kwargs={
                "queue": queue,
                "db_options": db_options,
                "start_param_df": params,
                "start_fitness": fitness_eval,
                "stop_signal": stop_signal,
            },
            daemon=False,
        )
        outer_server_process.start()

    result = _internal_minimize(
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        algorithm=algorithm,
        algo_options=algo_options,
        general_options=general_options,
        queue=queue,
        fitness_factor=fitness_factor,
    )

    if dashboard:
        stop_signal.set()
        outer_server_process.terminate()
    return result


def _one_argument_single_minimize(kwargs):
    """Wrapper for single_minimize used for multiprocessing.
    """
    return _single_minimize(**kwargs)


def _internal_minimize(
    criterion,
    criterion_kwargs,
    params,
    internal_params,
    constraints,
    algorithm,
    algo_options,
    general_options,
    queue,
    fitness_factor,
):
    """Create the internal criterion function and minimize it.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        params (pd.DataFrame):
            See :ref:`params`.

        internal_params (DataFrame):
            See :ref:`params`.

        constraints (list):
            list with constraint dictionaries. See for details.

        algorithm (str):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        algo_options (dict):
            algorithm specific configurations for the optimization

        general_options (dict):
            additional configurations for the optimization

        queue (Queue):
            queue to which the fitness evaluations and params DataFrames are supplied.

        fitness_factor (float):
            multiplicative factor for the fitness displayed in the dashboard.
            Set to -1 for maximizations to plot the fitness that is being maximized.

    """
    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        queue=queue,
        fitness_factor=fitness_factor,
    )

    current_dir_path = Path(__file__).resolve().parent
    with open(current_dir_path / "algo_dict.json") as j:
        algos = json.load(j)
    origin, algo_name = algorithm.split("_", 1)

    try:
        assert algo_name in algos[origin], "Invalid algorithm requested: {}".format(
            algorithm
        )
    except (AssertionError, KeyError):
        proposals = propose_algorithms(algorithm, algos)
        raise NotImplementedError(
            f"{algorithm} is not a valid choice. Did you mean one of {proposals}?"
        )

    if origin in ["nlopt", "pygmo"]:
        results = minimize_pygmo(
            internal_criterion, internal_params, params, origin, algo_name, algo_options
        )

    elif origin == "scipy":
        results = minimize_scipy(
            internal_criterion, internal_params, params, algo_name, algo_options
        )
    elif origin == "tao":
        results = minimize_pounders(
            internal_criterion, internal_params, criterion, params, algo_options
        )
    else:
        raise ValueError("Invalid algorithm requested.")

    params = reparametrize_from_internal(
        internal=results["x"],
        fixed_values=params["_internal_fixed_value"].to_numpy(),
        pre_replacements=params["_pre_replacements"].to_numpy().astype(int),
        processed_constraints=constraints,
        post_replacements=params["_post_replacements"].to_numpy().astype(int),
        processed_params=params,
    )

    return results, params


def create_internal_criterion(
    criterion, params, constraints, criterion_kwargs, queue, fitness_factor
):
    """Create the internal criterion function.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        constraints (list):
            list with constraint dictionaries. See for details.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        queue (Queue):
            queue to which the fitness evaluations and params DataFrames are supplied.

        fitness_factor (float):
            multiplicative factor for the fitness displayed in the dashboard.
            Set to -1 for maximizations to plot the fitness that is being maximized.

    Returns:
        internal_criterion (function):
            function that takes an internal_params DataFrame as only argument.
            It calls the original criterion function after the necessary
            reparametrizations and passes the results to the dashboard queue if given
            before returning the fitness evaluation.

    """
    c = np.zeros(1, dtype=int)

    def internal_criterion(x, counter=c):
        p = reparametrize_from_internal(
            internal=x,
            fixed_values=params["_internal_fixed_value"].to_numpy(),
            pre_replacements=params["_pre_replacements"].to_numpy().astype(int),
            processed_constraints=constraints,
            post_replacements=params["_post_replacements"].to_numpy().astype(int),
            processed_params=params,
        )
        fitness_eval = criterion(p, **criterion_kwargs)

        # For Pounders, return the sum of squared errors.
        _fitness_eval = (
            fitness_eval.sum() if isinstance(fitness_eval, np.ndarray) else fitness_eval
        )

        if queue is not None:
            queue.put(
                QueueEntry(
                    iteration=counter[0],
                    params=p,
                    fitness=fitness_factor * _fitness_eval,
                )
            )
        counter += 1
        return fitness_eval

    return internal_criterion


def _process_params(params):
    assert (
        not params.index.duplicated().any()
    ), "No duplicates allowed in the index of params."
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    else:
        params["lower"].fillna(-np.inf)

    if "upper" not in params.columns:
        params["upper"] = np.inf
    else:
        params["upper"].fillna(np.inf)

    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [index_element_to_string(tup) for tup in params.index]
        params["name"] = names

    assert "_fixed" not in params.columns, "Invalid column name _fixed in params_df."

    invalid_names = ["_fixed_value", "_is_fixed_to_value", "_is_fixed_to_other"]
    invalid_present_columns = []
    for col in params.columns:
        if col in invalid_names or col.startswith("_internal"):
            invalid_present_columns.append(col)

    if len(invalid_present_columns) > 0:
        msg = (
            "Column names starting with '_internal' and as well as any other of the "
            f"following columns are not allowed in params:\n{invalid_names}."
            f"This is violated for:\n{invalid_present_columns}."
        )
        raise ValueError(msg)
    return params
