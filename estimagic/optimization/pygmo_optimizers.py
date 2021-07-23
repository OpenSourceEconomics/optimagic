import functools

import numpy as np
import pygmo as pg

from estimagic import batch_evaluators
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS

POPULATION_SIZE = 10000
N_GENERATIONS = 500


def pygmo_gaco(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    population_size=POPULATION_SIZE,
    batch_evaluator=None,
    n_cores=1,
    seed=None,
    discard_start_params=False,
    #
    n_generations=N_GENERATIONS,
    kernel_size=63,
    convergence_speed=1.0,
    oracle=0.0,
    accuracy=0.01,
    threshold=1,
    std_convergence_speed=7,
    stopping_max_n_without_improvements=100000,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    focus=0.0,
    activate_memory_for_multiple_calls=False,
):
    """Minimize a scalar function using the extended ant colony algorithm.

    The version available through pygmo is an extended version of the original ant
    colony algorithm proposed by :cite:`Schlueter2009`.

    This algorithm can be applied to box-bounded problems.

    Ant colony optimization is a class of optimization algorithms modeled on the actions
    of an ant colony. Artificial ‘ants’ (e.g. simulation agents) locate optimal
    solutions by moving through a parameter space representing all possible solutions.
    Real ants lay down pheromones directing each other to resources while exploring
    their environment. The simulated ‘ants’ similarly record their positions and the
    quality of their solutions, so that in later simulation iterations more ants locate
    better solutions.

    The extended ant colony algorithm generates future generations of ants by using the
    a multi-kernel gaussian distribution based on three parameters (i.e., pheromone
    values) which are computed depending on the quality of each previous solution. The
    solutions are ranked through an oracle penalty method.

    - population_size (int): Size of the population.
    - batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      estimagic batch_evaluators. See :ref:`batch_evaluators`.
    - n_cores (int): Number of cores to use.
    - seed (int): seed used by the internal random number generator.
    - discard_start_params (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.

    - n_generations (int): Number of generations to evolve.
    - kernel_size (int): Number of solutions stored in the solution archive.
    - convergence_speed (float): This parameter is useful for managing the convergence
      speed towards the found minima (the smaller the faster).
    - oracle (float): oracle parameter used in the penalty method.
    - accuracy (float): accuracy parameter for maintaining a minimum penalty function's
      values distances.
    - threshold (int): when the generations reach the threshold then the convergence
      speed is set to 0.01 automatically.
    - std_convergence_speed (int): parameter that determines the convergence speed of
       the standard deviations.
    - stopping.max_n_without_improvements (int): if a positive integer is assigned here,
      the algorithm will count the runs without improvements, if this number exceeds the
      given value, the algorithm will be stopped.
    - stopping.max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      successful convergence.
    - focus (float): this parameter makes the search for the optimum greedier and more
      focused on local improvements (the higher the greedier). If the value is very
      high, the search is more focused around the current best solutions.
    - activate_memory_for_multiple_calls (bool): if true, memory is activated in the
      algorithm for multiple calls.

    """
    algo_options = {
        "population_size": population_size,
        "n_cores": n_cores,
        "seed": seed,
        "discard_start_params": discard_start_params,
    }
    if batch_evaluator is not None:
        algo_options["batch_evaluator"] = batch_evaluator

    algo_options.update(
        **{
            "gen": n_generations,
            "ker": kernel_size,
            "q": convergence_speed,
            "oracle": oracle,
            "acc": accuracy,
            "threshold": threshold,
            "n_gen_mark": std_convergence_speed,
            "impstop": stopping_max_n_without_improvements,
            "evalstop": stopping_max_criterion_evaluations,
            "focus": focus,
            "memory": activate_memory_for_multiple_calls,
        }
    )

    res = _minimize_pygmo(
        criterion_and_derivative=criterion_and_derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        method="gaco",
        algo_options=algo_options,
    )
    return res


def _minimize_pygmo(
    criterion_and_derivative, x, lower_bounds, upper_bounds, method, algo_options=None
):
    """Minimize a function with pygmo.

    Args:
        criterion_and_derivative (callable):
        x (np.ndarray): Starting values of the parameters.
        lower_bounds (np.ndarray):
        upper_bounds (np.ndarray):
        method (str): One of the optimizers of the pygmo package.
        algo_options (dict): Options for the optimizer. In addition to
            the algo options that will be passed directly to the pygmo
            algorithms we have the following entries:
            - population_size (int): Population size for genetic algorithms.
            - batch_evaluator (str or callable): An estimagic batch evaluator,
                default joblib batch evaluator.
            - n_cores (int): Number of cores used for parallel evaluation of
                the criterion function. Default 1.
            - seed (int or None): Random seed for drawing the initial
                population.
            - discard_start_params (bool): If True, the start params are not
                guaranteed to be part of the initial population. This saves one
                criterion function evaluation that cannot be done in parallel
                with other evaluations. Default False.

    Returns:
        results (dict): Dictionary with optimization results.

    """

    algo_options = {} if algo_options is None else algo_options.copy()
    population_size = algo_options.pop("population_size", POPULATION_SIZE)
    batch_evaluator = algo_options.pop("batch_evaluator", "joblib_batch_evaluator")
    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(batch_evaluators, batch_evaluator)
    n_cores = algo_options.pop("n_cores", 1)
    seed = algo_options.pop("seed", None)
    discard_start_params = algo_options.pop("discard_start_params", False)

    algo_info = {
        "parallelizes": n_cores != 1,
        "name": f"pygmo_{method}",
        "needs_scaling": True,
        "primary_criterion_entry": "value",
    }
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    bounds = (lower_bounds, upper_bounds)
    prob = _create_problem(
        func,
        bounds,
        gradient,
        dim=len(x),
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
    )
    algo = _create_algorithm(method, algo_options)
    pop = _create_population(
        prob, population_size, x, seed=seed, discard_start_params=discard_start_params
    )
    evolved = algo.evolve(pop)
    result = _process_pygmo_results(evolved)

    return result


def _create_problem(func, bounds, gradient_, dim, batch_evaluator, n_cores):
    class Problem:
        def fitness(self, x):
            return [func(x)]

        def get_bounds(self):
            return bounds

        def gradient(self, dv):
            return gradient_(dv)

        def batch_fitness(self, dvs):
            dv_list = dvs.reshape(-1, dim)
            eval_list = batch_evaluator(
                func=func,
                arguments=dv_list,
                n_cores=n_cores,
                # Error handling is done on a higher level
                error_handling="raise",
            )
            evals = np.array(eval_list)
            return evals

    problem = pg.problem(Problem())
    return problem


def _create_algorithm(method, algo_options):
    """Create a pygmo algorithm."""
    pygmo_uda = getattr(pg, method)
    algo = pygmo_uda(**algo_options)
    algo.set_bfe(pg.bfe())
    out = pg.algorithm(algo)
    return out


def _create_population(problem, population_size, x, seed, discard_start_params):
    """Create a pygmo population object.
    Args:
        problem (pygmo.Problem)
        algo_options (dict)
        x (np.ndarray)
    Todo:
        - constrain random initial values to be in some bounds
    """
    if not discard_start_params:
        population_size = population_size - 1

    pop = pg.population(
        problem,
        size=population_size,
        seed=seed,
        b=pg.bfe(),
    )
    if not discard_start_params:
        pop.push_back(x)
    return pop


def _process_pygmo_results(evolved):
    results = {
        # Harmonized results.
        "solution_x": evolved.champion_x,
        "solution_criterion": evolved.champion_f[0],
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": evolved.problem.get_fevals(),
        "n_derivative_evaluations": evolved.problem.get_gevals(),
        "n_iterations": None,
        "success": True,
        "reached_convergence_criterion": "Number of generations reached.",
        "message": None,
    }

    return results
