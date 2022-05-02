"""Collection of linear trust-region subsolvers."""
import math

import numpy as np


def minimize_trsbox_linear(
    linear_model, lower_bounds, upper_bounds, trustregion_radius, *, zero_treshold=1e-14
):
    """Minimize a linear trust-region subproblem using the trsbox algorithm.

    Solve the linear subproblem:

      min_x   g' * x
        s.t.   lower_bound <= x <= upper_bound
              ||x||^2 <= delta^2

    using an active-set approach.

    This algorithm is an implementation of the routine TRSBOX from
    M. J. D. Powell (2009) "The BOBYQA algorithm for bound constrained
    optimization without derivatives." (cite:`Powell2009`).

    Args:
        linear_model (namedtuple): Named tuple containing the parameters of the
            linear model, i.e. "linear_terms", which is a np.ndarray of shape (n,).
        lower_bound (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bound (np.ndarray): Upper bounds for x. Array of shape (n,).
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        (np.ndarray): Solution vector to the linear trust-region subproblem.
            Array of shape (n,).
    """
    lower_bounds_internal = np.minimum(lower_bounds, -zero_treshold)
    upper_bounds_internal = np.maximum(upper_bounds, zero_treshold)

    model_gradient = linear_model.linear_terms
    n = model_gradient.shape[0]
    x_candidate = np.zeros(n)

    direction = -model_gradient

    constant_directions = np.where(np.abs(direction) < zero_treshold)[0]
    direction[constant_directions] = 0.0

    active_directions = np.setdiff1d(np.arange(n), constant_directions)
    set_active_directions = iter(active_directions)

    for _ in range(n):

        if np.linalg.norm(direction) < zero_treshold:
            break

        x_candidate_unconstr = _take_unconstrained_step_up_to_boundary(
            x_candidate, direction, trustregion_radius, zero_treshold=zero_treshold
        )

        active_bound, index_active_bound = _find_next_active_bound(
            x_candidate_unconstr,
            lower_bounds_internal,
            upper_bounds_internal,
            set_active_directions,
        )

        if active_bound is None:
            x_candidate = x_candidate_unconstr
            break

        else:
            x_candidate, direction = _take_constrained_step_up_to_boundary(
                x_candidate,
                direction,
                active_bound,
                index_active_bound,
            )

    return x_candidate


def improve_geomtery_trsbox_linear(
    x_center,
    linear_model,
    lower_bounds,
    upper_bounds,
    trustregion_radius,
    *,
    zero_treshold=1e-14
):
    """Maximize a Lagrange polynomial of degree one to improve geometry of the model.

    Let a Lagrange polynomial of degree one be defined by:
        L(x) = c + g' * (x - x_center),

    where c and g denote the constant term and the linear terms (gradient)
    of the linear model, respectively.

    In order to maximize L(x), we maximize the absolute value of L(x) in a
    trust-region setting. I.e. we solve:
        max_x  abs(c + g' * (x - x_center))
            s.t. lower_bound <= x <= upper_bound
                 ||x - x_center|| <= delta

    In order to find x*, we both minimize and maximize g' * (x - center), respectively.
    The resulting candidate vectors are then plugged into the objective function L(x)
    to see which one yields the largest absolute value of the Lagrange polynomial.

    Args:
        x_center (np.ndarray): Center for the candidate vector x of shape (n,).
        linear_model (namedtuple): Named tuple containing the parameters of the
            linear model that form the Lagrange polynomial, including:
            - "constant_term", which is a floating point number, and
            - "linear_terms", which is a np.ndarray of shape (n,).
        lower_bounds (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bounds (np.ndarray): Upper bounds for x. Array of shape (n,).
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        (np.ndarray): Vector of shape (n,) that maximizes the Lagrange polynomial.
    """
    # Check if bounds valid
    if np.any(lower_bounds > x_center + zero_treshold):
        raise ValueError("x_base violates lower bound.")
    if np.any(x_center - zero_treshold > upper_bounds):
        raise ValueError("x_base violates upper bound.")

    linear_model_to_minimize = linear_model
    linear_model_to_maximize = linear_model._replace(
        linear_terms=-linear_model.linear_terms
    )

    # Minimize and maximize g' * (x - x_center), respectively
    x_candidate_min = minimize_trsbox_linear(
        linear_model_to_minimize,
        lower_bounds - x_center,
        upper_bounds - x_center,
        trustregion_radius,
        zero_treshold=zero_treshold,
    )
    x_candidate_max = minimize_trsbox_linear(
        linear_model_to_maximize,
        lower_bounds - x_center,
        upper_bounds - x_center,
        trustregion_radius,
        zero_treshold=zero_treshold,
    )

    lagrange_polynomial = lambda x: abs(
        linear_model.constant_term + np.dot(linear_model.linear_terms, x)
    )

    if lagrange_polynomial(x_candidate_min) >= lagrange_polynomial(x_candidate_max):
        x_lagrange = x_candidate_min + x_center
    else:
        x_lagrange = x_candidate_max + x_center

    return x_lagrange


def _find_next_active_bound(
    x_candidate_unconstr,
    lower_bounds,
    upper_bounds,
    set_active_directions,
):
    """Find the next active bound and return its index.

    A (lower or upper) bound is considered active if
        x_candidate <= lower_bounds
        x_candidate >= upper_bounds

    Args:
        x_candidate_unconstr (np.ndarray): Unconstrained candidate vector of shape (n,),
            which has been computed without taking bounds into account.
        lower_bounds (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bounds (np.ndarray): Upper bounds for x. Array of shape (n,).
        set_active_directions (iterator): Iterator over the indices of active search
            directions, i.e. directions that are not zero.

    Returns:
        (tuple):
            - active_bound (float or None): The next active bound. It can be a lower
                or active bound. If None, there are no more active bounds left in the
                set of active search directions.
            - index_bound_active (int or None): Index where an active lower or
                upper bound has been found. None, if no active bound has been detected.
    """
    index_active = next(set_active_directions)

    while True:
        if x_candidate_unconstr[index_active] >= upper_bounds[index_active]:
            active_bound = upper_bounds[index_active]
            break

        elif x_candidate_unconstr[index_active] <= lower_bounds[index_active]:
            active_bound = lower_bounds[index_active]
            break

        else:
            try:
                index_active = next(set_active_directions)
            except StopIteration:
                active_bound = None
                break

    return active_bound, index_active


def _take_constrained_step_up_to_boundary(
    x_candidate, direction, active_bound, index_bound_active
):
    """Take largest constrained step possible until trust-region boundary is hit.

    Args:
        x_candidate (np.ndarray): Current candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        active_bound (float): The active - lower or upper - bound.
        index_bound_active (int): Index where an active lower or upper bound
            has been found.

    Returns:
        (tuple):
        - x_candidate (np.ndarray): New candidate vector of shape (n,).
        - direction (np.ndarray): New direction vector of shape (n,), where the
            search direction of the currently active bound has been set to zero.
    """
    direction_updated = np.copy(direction)

    step_size_constr = (active_bound - x_candidate[index_bound_active]) / direction[
        index_bound_active
    ]

    x_candidate = x_candidate + step_size_constr * direction
    x_candidate[index_bound_active] = active_bound

    # Do not search in this direction anymore
    direction_updated[index_bound_active] = 0.0

    return x_candidate, direction_updated


def _take_unconstrained_step_up_to_boundary(
    x_candidate, direction, trustregion_radius, zero_treshold
):
    """Take largest unconstrained step possible until trust-region boundary is hit.

    Args:
        x_candidate (np.ndarray): Current candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        (np.ndarray): New unconstrained candidate vector shape (n,).
    """
    step_size_unconstr = _get_distance_to_trustregion_boundary(
        x_candidate, direction, trustregion_radius, zero_treshold
    )
    x_candidate_unconstr = x_candidate + step_size_unconstr * direction

    return x_candidate_unconstr


def _get_distance_to_trustregion_boundary(x0, direction, radius, zero_treshold):
    """Compute the candidate vector's distance to the trustregion boundary.

    Given the candidate vector, find the largest step `alpha` in direction `g`
    that satisfies ||x|| <= delta,

    where `g` denotes the direction vector and `delta` the trust-region radius.

    To find `alpha`, i.e. the candidate's distance to the trust-region boundary, solve
      ||x0 + alpha * g||^2 = delta^2
         s.t. alpha >= 0

    Using this method, the solution exists whenever ||x0|| <= delta^2.

    Choose alpha = 0, if the direction vector is zero everywhere.

    Args:
        x0 (np.ndarray): Candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        (float) Distance of the candidate vector to the trustregion
            boundary.
    """
    g_dot_x0 = np.dot(direction, x0)
    g_sumsq = np.dot(direction, direction)
    x0_sumsq = np.dot(x0, x0)

    if math.sqrt(g_sumsq) < zero_treshold:
        alpha = 0
    else:
        alpha = (
            math.sqrt(np.maximum(0, g_dot_x0**2 + g_sumsq * (radius**2 - x0_sumsq)))
            - g_dot_x0
        ) / g_sumsq

    return alpha