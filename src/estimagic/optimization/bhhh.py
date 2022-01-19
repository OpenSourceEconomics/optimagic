from functools import partial

import numpy as np


def bhhh(
    criterion_and_derivative,
    x,
    convergence_absolute_gradient_tolerance=1e-8,
    stopping_max_iterations=200,
):
    """
    Minimize a likelihood function using the BHHH algorithm.

    For details,
    """
    algorithm_info = {
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": False,
        "name": "bhhh",
    }
    _criterion_and_derivative = partial(
        criterion_and_derivative, algorithm_info=algorithm_info
    )

    result_dict = bhhh_internal(
        criterion_and_derivative=_criterion_and_derivative,
        x=x,
        convergence_absolute_gradient_tolerance=convergence_absolute_gradient_tolerance,
        stopping_max_iterations=stopping_max_iterations,
    )

    return result_dict


def bhhh_internal(
    criterion_and_derivative,
    x,
    convergence_absolute_gradient_tolerance,
    stopping_max_iterations,
):
    """
    Minimize scalar function of one or more variables via the BHHH algorithm.
    Args:
        criterion_and_derivative (callable): The objective function to be minimized.
        x (np.ndarray): Initial guess. Array of real elements of size (n,),
            where `n` is the number of parameters.
        convergence_absolute_gradient_tolerance (float): Tolerance for termination.
        stopping_max_iterations (int): Maximum number of iterations to perform.

    Returns:
        (dict) Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n,).
        - solution_criterion (np.ndarray): Values of the criterion function at the
            solution vector. Shape (n_obs,).
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution vector or reaching maxiter.
        - message (str): Message to the user. Currently it says: "Under development."
    """
    criterion_accepted, derivative = criterion_and_derivative(
        x, task="criterion_and_derivative"
    )
    x_accepted = x

    hessian_approx = np.dot(derivative.T, derivative)
    gradient = np.sum(derivative, axis=0)
    direction = np.linalg.solve(hessian_approx, gradient)
    gtol = np.dot(gradient, direction)

    initial_step_size = 1
    step_size = initial_step_size

    niter = 1
    while niter < stopping_max_iterations:
        niter += 1

        x_candidate = x_accepted + step_size * direction
        criterion_candidate = criterion_and_derivative(x_candidate, task="criterion")

        # If previous step was accepted
        if step_size == initial_step_size:
            derivative = criterion_and_derivative(x_candidate, task="derivative")
            hessian_approx = np.dot(derivative.T, derivative)

        else:
            criterion_candidate, derivative = criterion_and_derivative(
                x_candidate, task="criterion_and_derivative"
            )

        # Line search
        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            step_size /= 2

            if step_size <= 0.01:
                # Accept step
                x_accepted = x_candidate
                criterion_accepted = criterion_candidate

                # Reset step size
                step_size = initial_step_size

        # If decrease in likelihood, calculate new direction vector
        else:
            # Accept step
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate

            gradient = np.sum(derivative, axis=0)
            direction = np.linalg.solve(hessian_approx, gradient)
            gtol = np.dot(gradient, direction)

            if gtol < 0:
                hessian_approx = np.dot(derivative.T, derivative)
                direction = np.linalg.solve(hessian_approx, gradient)

            # Reset stepsize
            step_size = initial_step_size

        if gtol < convergence_absolute_gradient_tolerance:
            break

    result_dict = {
        "solution_x": x_accepted,
        "solution_criterion": criterion_accepted,
        "n_iterations": niter,
        "message": "Under develpment",
    }

    return result_dict
