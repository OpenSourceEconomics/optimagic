import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.differentiation.numdiff_np import _consolidate_one_step_central
from estimagic.differentiation.numdiff_np import _consolidate_one_step_forward
from estimagic.differentiation.numdiff_np import _fill_nans_with_other
from estimagic.differentiation.numdiff_np import _get_output_shape
from estimagic.differentiation.numdiff_np import _nan_skipping_batch_evaluator
from estimagic.differentiation.numdiff_np import jacobian
from estimagic.examples.numdiff_example_functions_np import logit_loglikeobs
from estimagic.examples.numdiff_example_functions_np import logit_loglikeobs_jacobian


@pytest.fixture
def binary_choice_inputs():
    fix_path = Path(__file__).resolve().parent / "binary_choice_inputs.pickle"
    with open(fix_path, "rb") as p:
        inputs = pickle.load(p)
    return inputs


methods = ["forward", "backward", "central"]


@pytest.mark.parametrize("method", methods)
def test_jacobian(binary_choice_inputs, method):
    fix = binary_choice_inputs
    func = partial(logit_loglikeobs, y=fix["y"], x=fix["x"])

    calculated = jacobian(
        func=func,
        method=method,
        x=fix["params_np"],
        n_steps=1,
        base_steps=None,
        lower_bounds=np.full(fix["params_np"].shape, -np.inf),
        upper_bounds=np.full(fix["params_np"].shape, np.inf),
        min_steps=1e-8,
        step_ratio=2.0,
        f0=func(fix["params_np"]),
        n_processes=1,
    )

    expected = logit_loglikeobs_jacobian(fix["params_np"], fix["y"], fix["x"])

    aaae(calculated, expected, decimal=6)


def test_fill_nans_with_other():
    a = np.array([[1, np.nan], [np.nan, 3]])
    b = np.array([[5, 5], [3, 4]])
    calculated = _fill_nans_with_other(a, b)
    expected = np.array([[1, 5], [3, 3]])
    aaae(calculated, expected)


def test_get_output_shape():
    a = [np.nan, 7, np.ones((3, 4)), 5]
    assert _get_output_shape(a) == (3, 4)


def test_nan_skipping_batch_evaluator():
    arglist = [np.nan, np.ones(2), np.array([3, 4]), np.nan, np.array([1, 2])]
    expected = [
        np.full(2, np.nan),
        np.ones(2),
        np.array([9, 16]),
        np.full(2, np.nan),
        np.array([1, 4]),
    ]
    calculated = _nan_skipping_batch_evaluator(lambda x: x ** 2, arglist, 1)
    for arr_calc, arr_exp in zip(calculated, expected):
        if np.isnan(arr_exp).all():
            assert np.isnan(arr_calc).all()
        else:
            aaae(arr_calc, arr_exp)


def test_consolidate_one_step_forward():
    forward = np.ones((1, 4, 3))
    forward[:, :, 0] = np.nan
    backward = np.zeros_like(forward)

    calculated = _consolidate_one_step_forward(
        {"forward": forward, "backward": backward}
    )
    expected = np.array([[0, 1, 1]] * 4)
    aaae(calculated, expected)


def test_consolidate_one_step_central():
    central = np.full((1, 4, 3), np.nan)
    central[:, :, 2] = 3
    forward = np.ones((1, 4, 3))
    forward[:, :, 0] = np.nan
    backward = np.zeros_like(forward)
    backward[:, :, 1] = np.nan

    expected = np.array([[0, 1, 3]] * 4)
    calculated = _consolidate_one_step_central(
        {"forward": forward, "backward": backward, "central": central}
    )
    aaae(calculated, expected)
