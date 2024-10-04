import itertools
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic import maximize, minimize
from optimagic.algorithms import NonlinearConstrainedAlgorithms
from optimagic.config import IS_CYIPOPT_INSTALLED
from optimagic.parameters.bounds import Bounds

NLC_ALGORITHMS = NonlinearConstrainedAlgorithms()._available_algorithms_dict

# ======================================================================================
# Two-dimension example with equality and inequality constraints
# ======================================================================================


@pytest.fixture()
def nlc_2d_example():
    """Non-linear constraints: 2-dimensional example.

    See the example section in https://en.wikipedia.org/wiki/Nonlinear_programming.

    """

    def criterion(x):
        return np.sum(x)

    def derivative(x):
        return np.ones_like(x)

    def constraint_func(x):
        value = np.dot(x, x)
        return np.array([value - 1, 2 - value])

    def constraint_jac(x):
        return 2 * np.row_stack((x.reshape(1, -1), -x.reshape(1, -1)))

    constraints_long = om.NonlinearConstraint(
        func=constraint_func,
        derivative=constraint_jac,
        lower_bound=np.zeros(2),
        tol=1e-8,
    )

    constraints_flat = om.NonlinearConstraint(
        func=lambda x: np.dot(x, x),
        derivative=lambda x: 2 * x,
        lower_bound=1,
        upper_bound=2,
        tol=1e-8,
    )

    constraints_equality = om.NonlinearConstraint(
        func=lambda x: np.dot(x, x),
        derivative=lambda x: 2 * x,
        value=2,
    )

    constraints_equality_and_inequality = [
        om.NonlinearConstraint(
            func=lambda x: np.dot(x, x),
            derivative=lambda x: 2 * x,
            lower_bound=1,
        ),
        om.NonlinearConstraint(
            func=lambda x: np.dot(x, x),
            derivative=lambda x: 2 * x,
            value=2,
        ),
    ]

    _kwargs = {
        "criterion": criterion,
        "params": np.array([0, np.sqrt(2)]),
        "derivative": derivative,
        "bounds": Bounds(lower=np.zeros(2), upper=2 * np.ones(2)),
    }

    kwargs = {
        "flat": {**_kwargs, "constraints": constraints_flat},
        "long": {**_kwargs, "constraints": constraints_long},
        "equality": {**_kwargs, "constraints": constraints_equality},
        "equality_and_inequality": {
            **_kwargs,
            "constraints": constraints_equality_and_inequality,
        },
    }

    solution_x = np.ones(2)

    return solution_x, kwargs


TEST_CASES = list(
    itertools.product(
        NLC_ALGORITHMS, ["flat", "long", "equality", "equality_and_inequality"]
    )
)


@pytest.mark.parametrize("algorithm, constr_type", TEST_CASES)
def test_nonlinear_optimization(nlc_2d_example, algorithm, constr_type):
    """Test that available nonlinear optimizers solve a nonlinear constraints problem.

    We test for the cases of "equality", "inequality" and "equality_and_inequality"
    constraints.

    """
    if "equality" in constr_type and algorithm == "nlopt_mma":
        pytest.skip(reason="Very slow and low accuracy.")

    solution_x, kwargs = nlc_2d_example
    if algorithm == "scipy_cobyla":
        del kwargs[constr_type]["bounds"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = maximize(algorithm=algorithm, **kwargs[constr_type])

    if NLC_ALGORITHMS[algorithm].__algo_info__.is_global:
        decimal = 0
    else:
        decimal = 4

    aaae(result.params, solution_x, decimal=decimal)


# ======================================================================================
# Documentation example
# ======================================================================================


def criterion(params):
    offset = np.linspace(1, 0, len(params))
    x = params - offset
    return x @ x


@pytest.mark.parametrize("algorithm", NLC_ALGORITHMS)
def test_documentation_example(algorithm):
    if algorithm in ("nlopt_mma", "ipopt"):
        pytest.skip(reason="Slow.")

    kwargs = {
        "bounds": Bounds(lower=np.zeros(6), upper=2 * np.ones(6)),
    }

    if algorithm == "scipy_cobyla":
        del kwargs["bounds"]

    minimize(
        fun=criterion,
        params=np.ones(6),
        algorithm=algorithm,
        constraints=om.NonlinearConstraint(
            func=lambda x: np.prod(x),
            selector=lambda x: x[:-1],
            value=1.0,
        ),
        **kwargs,
    )


# ======================================================================================
# Test: selection + reparametrization constraint + nonlinear constraint
# ======================================================================================


@pytest.fixture()
def general_example():
    params = {"a": np.array([0.1, 0.3, 0.4, 0.2]), "b": np.array([1.5, 2])}

    def criterion(params):
        weights = np.array([0, 1, 2, 3])
        return params["a"] @ weights + params["b"].sum()

    def selector_probability_constraint(params):
        return params["a"]

    def selector_nonlinear_constraint(params):
        return {"probs": params["a"][:3][::-1], "unnecessary": params["b"]}

    def constraint(selected):
        return selected["probs"] @ selected["probs"]

    constraints = [
        om.ProbabilityConstraint(
            selector=selector_probability_constraint,
        ),
        om.NonlinearConstraint(
            selector=selector_nonlinear_constraint,
            upper_bound=0.8,
            func=constraint,
            tol=0.01,
        ),
        om.NonlinearConstraint(
            selector=selector_nonlinear_constraint,
            func=constraint,
            upper_bound=0.8,
            tol=0.01,
        ),
    ]

    lower_bound = {"b": np.array([0, 0])}
    upper_bound = {"b": np.array([2, 2])}

    kwargs = {
        "fun": criterion,
        "params": params,
        "constraints": constraints,
        "lower_bounds": lower_bound,
        "upper_bounds": upper_bound,
    }
    return kwargs


TEST_CASES = list(itertools.product(["ipopt"], [True, False]))


@pytest.mark.skipif(not IS_CYIPOPT_INSTALLED, reason="Needs ipopt")
@pytest.mark.parametrize("algorithm, skip_checks", TEST_CASES)
def test_general_example(general_example, algorithm, skip_checks):
    kwargs = general_example

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(algorithm=algorithm, skip_checks=skip_checks, **kwargs)

    optimal_p1 = 0.5 + np.sqrt(3 / 20)  # can be derived analytically
    optimal_p2 = 1 - optimal_p1

    aaae(res.params["a"], np.array([optimal_p1, optimal_p2, 0, 0]), decimal=4)
    aaae(res.params["b"], np.array([0.0, 0]), decimal=5)
