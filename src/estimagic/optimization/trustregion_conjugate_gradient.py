"""Implementation of the Steihaug-Toint Conjugate Gradient algorithm."""
from math import sqrt

import numpy as np


def minimize_trust_cg(model_gradient, model_hessian, trustregion_radius):
    """Minimize the quadratic trust-region subproblem using Conjugate Gradient method.

    This algorithm is an implementation of the Steihaug-Toint truncated
    Conjugate Gradient algorithm as described in

        T. Steihaug (1983), "The conjugate gradient method and trust regions in large
            scale optimization", SIAM Journal on Numerical Analysis 20 (3),
            pp. 626-637

    Args:
        model_gradient (np.ndarray): Gradient of the quadratic model. Shape (n,).
        model_hessian (np.ndarray): Hessian of the quadratic model. Shape (n, n).
        trustregion_radius (float): Radius of the trust-region.

    Returns:
        (np.ndarray): Solution vector to the quadratic trust-region subproblem
            with shape (n,).
    """
    n = model_gradient.shape[0]
    maxiter = 2 * n

    abstol = 1.0e-8
    reltol = 1.0e-6

    residual = np.copy(model_gradient)
    x_candidate = np.zeros(n)

    # Use steepest descent direction at initial point
    direction = -residual

    gnorm = np.linalg.norm(residual)
    stop_tol = max(abstol, reltol * gnorm)

    niter = 0
    while gnorm > stop_tol and niter < maxiter:
        niter += 1

        square_term = np.dot(np.dot(direction, model_hessian), direction)

        distance_to_boundary = _get_distance_to_trustregion_boundary(
            x_candidate, direction, trustregion_radius
        )

        # Length of the Conjugate Gradient step
        alpha = np.dot(residual, residual) / square_term

        if square_term <= 0 or alpha > distance_to_boundary:
            x_candidate = x_candidate + distance_to_boundary * direction
            break

        x_candidate, residual, direction = _update_vectors_for_next_iteration(
            x_candidate, residual, direction, model_hessian, alpha
        )
        gnorm = np.linalg.norm(residual)

    return x_candidate


def _update_vectors_for_next_iteration(
    x_candidate, residual, direction, hessian, alpha
):
    """Update candidate, residual, and direction vectors for the next iteration.

    Args:
        x_candidate (np.ndarray): Candidate vector of shape (n,).
        residual (np.ndarray): Array of residuals of shape (n,). The residual vector
            is defined as `r = Ax - b`, where `A` denotes the hessian matrix and `b` the
            gradient vector of the quadratic trust-region subproblem.
            `r` is equivalent to the first derivative of the quadratic subproblem.
        direction (np.ndarray): Direction vector of shape (n,).

    Returns:
        (tuple) Tuple containing:
            - x_candidate (np.ndarray): Updated candidate vector of shape (n,).
            - residual (np.ndarray): Updated array of residuals of shape (n,).
            - direction (np.darray): Updated direction vector of shape (n,).
    """
    residual_old = np.copy(residual)

    x_candidate = x_candidate + alpha * direction
    residual = residual + alpha * np.dot(hessian, direction)

    beta = np.dot(residual, residual) / np.dot(residual_old, residual_old)
    direction = -residual + beta * direction

    return x_candidate, residual, direction


def _get_distance_to_trustregion_boundary(candidate, direction, radius):
    """Compute the distance of the candidate vector to trustregion boundary.

    The positive distance sigma is defined in Eculidean norm, as follows:

        `|| s + sigma * p || = radius`

    where `s` denotes the candidate vector, and `p` the direction vector.

    Args:
        candidate(np.ndarray): Candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        radius (floar): Radius of the trust-region

    Returns:
        (float) Distance of the candidate vector to the trustregion
            boundary.
    """
    ss = np.dot(candidate, candidate)
    sp = np.dot(candidate, direction)
    pp = np.dot(direction, direction)

    sigma = -sp + sqrt(sp * sp + pp * (radius**2 - ss))
    sigma /= pp

    return sigma
