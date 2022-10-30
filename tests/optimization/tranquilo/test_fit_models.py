import numpy as np
import pytest
import yaml
from estimagic import first_derivative
from estimagic import second_derivative
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.tranquilo.fit_models import _get_current_fit_pounders
from estimagic.optimization.tranquilo.fit_models import _get_feature_matrices_pounders
from estimagic.optimization.tranquilo.fit_models import (
    _interactions_and_square_features,
)
from estimagic.optimization.tranquilo.fit_models import _polynomial_features
from estimagic.optimization.tranquilo.fit_models import _reshape_square_terms_to_hess
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import ModelInfo
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def aaae(x, y, case=None):
    tolerance = {
        None: 3,
        "hessian": 2,
        "gradient": 3,
    }
    assert_array_almost_equal(x, y, decimal=tolerance[case])


def read_yaml(path):
    with open(rf"{path}") as file:
        data = yaml.full_load(file)

    return data


def _transform_square_terms_to_pounders(square_terms, n_params, n_residuals):

    for k in range(n_residuals):
        num = 0
        for i in range(n_params):
            square_terms[k, i, i] *= 0.5
            num += 1
            for j in range(i + 1, n_params):
                square_terms[k, j, i] /= np.sqrt(2)
                square_terms[k, i, j] /= np.sqrt(2)
                num += 1

    return square_terms


# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture
def quadratic_case():
    """Test scenario with true quadratic function.

    We return true function, and function evaluations and data on random points.
    """
    n_params = 4
    n_samples = 2_000

    # theoretical terms
    linear_terms = 1 + np.arange(n_params)
    square_terms = np.arange(n_params**2).reshape(n_params, n_params)
    square_terms = square_terms + square_terms.T

    def func(x):
        y = -10 + linear_terms @ x + 0.5 * x.T @ square_terms @ x
        return y

    x0 = np.ones(n_params)

    # random data
    x = np.array(
        [x0 + np.random.uniform(-0.01 * x0, 0.01 * x0) for _ in range(n_samples)]
    )
    y = np.array([func(_x) for _x in list(x)]).reshape(-1, 1)

    out = {
        "func": func,
        "x0": x0,
        "x": x,
        "y": y,
        "linear_terms_expected": linear_terms,
        "square_terms_expected": square_terms,
    }
    return out


@pytest.fixture
def just_identified_case():
    """Test scenario with true quadratic function and n + 1 points.

    We return true function, and function evaluations and data on random points.
    """
    n_params = 4
    n_samples = n_params + 1

    # theoretical terms
    linear_terms = 1 + np.arange(n_params)
    square_terms = np.zeros((n_params, n_params))

    def func(x):
        y = -10 + linear_terms @ x + 0.5 * x.T @ square_terms @ x
        return y

    x0 = np.ones(n_params)

    # random data
    x = np.array(
        [x0 + np.random.uniform(-0.01 * x0, 0.01 * x0) for _ in range(n_samples)]
    )
    y = np.array([func(_x) for _x in list(x)]).reshape(-1, 1)

    out = {
        "func": func,
        "x0": x0,
        "x": x,
        "y": y,
        "linear_terms_expected": linear_terms,
        "square_terms_expected": square_terms,
    }
    return out


@pytest.fixture
def data_fit_pounders():
    """Test data from Tao Pounders."""
    test_data = read_yaml(TEST_FIXTURES_DIR / "get_coefficients_residual_model.yaml")

    n_params = 3
    n_samples = 2 * n_params + 1
    n_poly_features = n_params * (n_params + 1) // 2
    _is_just_identified = False

    inputs_dict = {
        "y": np.array(test_data["f_interpolated"]),
        "m_mat": np.array(test_data["x_sample_monomial_basis"])[
            : n_params + 1, : n_params + 1
        ],
        "n_mat": np.array(test_data["monomial_basis"])[:n_samples],
        "z_mat": np.array(test_data["basis_null_space"]),
        "n_z_mat": np.array(test_data["lower_triangular"])[:, n_params + 1 : n_samples],
        "n_params": n_params,
        "n_poly_features": n_poly_features,
        "has_intercepts": False,
        "_is_just_identified": _is_just_identified,
    }

    expected = {
        "linear_terms": np.array(test_data["linear_terms_expected"]),
        "square_terms": np.array(test_data["square_terms_expected"]),
    }

    return inputs_dict, expected


@pytest.fixture
def data_get_feature_matrices_pounders():
    test_data = read_yaml(
        TEST_FIXTURES_DIR / "get_interpolation_matrices_residual_model.yaml"
    )

    n_params = 3
    n_samples = 2 * n_params + 1
    n_poly_features = n_params * (n_params + 1) // 2
    center = np.array(test_data["x_accepted"])
    radius = test_data["delta"]

    model_indices = np.array([13, 12, 11, 10, 9, 8, 6])
    history_x = np.array(test_data["history_x"])
    x = (history_x[model_indices] - center) / radius

    inputs_dict = {
        "x": x,
        "model_indices": np.array(test_data["model_indices"]),
        "n_params": n_params,
        "n_samples": n_samples,
        "n_poly_features": n_poly_features,
    }

    expected = {
        "m_mat": np.array(test_data["x_sample_monomial_basis_expected"])[
            : n_params + 1, : n_params + 1
        ],
        "n_mat": np.array(test_data["monomial_basis_expected"]),
        "z_mat": np.array(test_data["basis_null_space_expected"]),
        "n_z_mat": np.array(test_data["lower_triangular_expected"])[
            :, n_params + 1 : n_samples
        ],
    }

    return inputs_dict, expected


# ======================================================================================
# Tests
# ======================================================================================


def test_fit_ols_against_truth(quadratic_case):
    fit_ols = get_fitter("ols")
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    aaae(got.linear_terms.squeeze(), quadratic_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), quadratic_case["square_terms_expected"])


@pytest.mark.parametrize("scenario", ["just_identified_case", "quadratic_case"])
def test_fit_pounders_against_truth(scenario, request):
    test_case = request.getfixturevalue(scenario)

    model_info = ModelInfo(has_intercepts=True, has_squares=True, has_interactions=True)
    fit_pounders = get_fitter("pounders", model_info=model_info)
    got = fit_pounders(test_case["x"], test_case["y"])

    aaae(got.linear_terms.squeeze(), test_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), test_case["square_terms_expected"])


@pytest.mark.parametrize("scenario", ["just_identified_case", "quadratic_case"])
def test_fit_pounders_experimental_against_truth(scenario, request):
    test_case = request.getfixturevalue(scenario)

    model_info = ModelInfo(has_intercepts=True, has_squares=True, has_interactions=True)
    fit_pounders = get_fitter("_pounders_experimental", model_info=model_info)
    got = fit_pounders(test_case["x"], test_case["y"])

    aaae(got.linear_terms.squeeze(), test_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), test_case["square_terms_expected"])


@pytest.mark.parametrize("scenario", ["just_identified_case", "quadratic_case"])
def test_pounders_no_intercepts(scenario, request):

    test_case = request.getfixturevalue(scenario)

    model_info = ModelInfo(
        has_intercepts=False, has_squares=True, has_interactions=True
    )
    fit_pounders = get_fitter("pounders", model_info=model_info)
    got = fit_pounders(test_case["x"], test_case["y"])

    assert got.intercepts is None
    aaae(got.linear_terms.squeeze(), test_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), test_case["square_terms_expected"])


@pytest.mark.parametrize("scenario", ["just_identified_case", "quadratic_case"])
def test_pounders_experimental_no_intercepts(scenario, request):

    test_case = request.getfixturevalue(scenario)

    model_info = ModelInfo(
        has_intercepts=False, has_squares=True, has_interactions=True
    )
    fit_pounders = get_fitter("_pounders_experimental", model_info=model_info)
    got = fit_pounders(test_case["x"], test_case["y"])

    assert got.intercepts is None
    aaae(got.linear_terms.squeeze(), test_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), test_case["square_terms_expected"])


@pytest.mark.parametrize("model", ["ols", "ridge"])
def test_fit_ols_against_gradient(model, quadratic_case):
    if model == "ridge":
        options = {"l2_penalty_square": 0}
    else:
        options = None

    fit_ols = get_fitter(model, options)
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    a = got.linear_terms.squeeze()
    hess = got.square_terms.squeeze()
    grad = a + hess @ quadratic_case["x0"]

    gradient = first_derivative(quadratic_case["func"], quadratic_case["x0"])
    aaae(gradient["derivative"], grad, case="gradient")


@pytest.mark.parametrize(
    "model, options",
    [("ols", None), ("ridge", {"l2_penalty_linear": 0, "l2_penalty_square": 0})],
)
def test_fit_ols_against_hessian(model, options, quadratic_case):
    fit_ols = get_fitter(model, options)
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])
    hessian = second_derivative(quadratic_case["func"], quadratic_case["x0"])
    hess = got.square_terms.squeeze()
    aaae(hessian["derivative"], hess, case="hessian")


@pytest.mark.parametrize("has_intercepts, has_squares", [(True, False), (True, False)])
def test_polynomial_features(has_intercepts, has_squares):

    x = np.array([[0, 1, 2], [3, 4, 5]])

    expected = {
        # (has_intercepts, has_squares): expected value,
        (True, True): np.array(
            [[1, 0, 1, 2, 0, 0, 0, 1, 2, 4], [1, 3, 4, 5, 9, 12, 15, 16, 20, 25]]
        ),
        (True, False): np.array([[1, 0, 1, 2, 0, 0, 2], [1, 3, 4, 5, 12, 15, 20]]),
        (False, True): np.array(
            [[0, 1, 2, 0, 0, 0, 1, 2, 4], [3, 4, 5, 9, 12, 15, 16, 20, 25]]
        ),
        (False, False): np.array([[0, 1, 2, 0, 0, 2], [3, 4, 5, 12, 15, 20]]),
    }

    got = _polynomial_features(
        x, has_intercepts=has_intercepts, has_squares=has_squares
    )

    assert_array_equal(got, expected[(has_intercepts, has_squares)])


@pytest.mark.parametrize("has_squares", [False, True])
def test_square_features_pounders(has_squares):
    has_intercepts = False

    x = np.array([[0, 1, 2], [3, 4, 5]])

    expected = {
        # (has_intercepts, has_squares): expected value,
        (False, True): np.array(
            [[0, 1, 2, 0, 0, 0, 1, 2, 4], [3, 4, 5, 9, 12, 15, 16, 20, 25]]
        ),
        (False, False): np.array([[0, 1, 2, 0, 0, 2], [3, 4, 5, 12, 15, 20]]),
    }

    got_interactions_and_square_features = _interactions_and_square_features(
        x, has_squares=has_squares
    )
    polynomial_features = np.concatenate(
        (x, got_interactions_and_square_features), axis=1
    )

    assert_array_equal(polynomial_features, expected[(has_intercepts, has_squares)])


def test_polynomial_feature_matrices_pounders(
    data_get_feature_matrices_pounders,
):
    inputs, expected = data_get_feature_matrices_pounders

    x = inputs["x"]
    model_info = ModelInfo(has_intercepts=True, has_squares=True, has_interactions=True)

    m_mat, _n_mat, z_mat, _n_z_mat = _get_feature_matrices_pounders(x, model_info)

    assert_array_almost_equal(m_mat, expected["m_mat"])
    assert_array_almost_equal(z_mat, expected["z_mat"])


def test_get_current_fit_pounders_transform_after_reshaping(data_fit_pounders):
    inputs, expected = data_fit_pounders
    n_params = inputs["n_params"]
    has_squares = True
    n_residuals = 214

    coef = _get_current_fit_pounders(**inputs)
    linear_terms, _square_terms = np.split(coef, (n_params,), axis=1)

    _square_terms = _reshape_square_terms_to_hess(
        _square_terms, n_params, n_residuals, has_squares
    )
    square_terms = _transform_square_terms_to_pounders(
        _square_terms, n_params, n_residuals
    )

    assert_array_almost_equal(linear_terms, expected["linear_terms"])
    assert_array_almost_equal(square_terms, expected["square_terms"])
