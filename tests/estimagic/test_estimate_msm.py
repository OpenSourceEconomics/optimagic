"""Most test exploit the special case where simulate_moments just returns parameters."""

import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal

from estimagic.estimate_msm import estimate_msm
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.optimizers import scipy_optimizers
from optimagic.shared.check_option_dicts import (
    check_optimization_options,
)


def _sim_pd(params):
    return pd.Series(params)


def _sim_np(params):
    return params


def _sim_dict_pd(params):
    return {"simulated_moments": pd.Series(params), "other": "bla"}


def _sim_dict_np(params):
    return {"simulated_moments": params, "other": "bla"}


cov_np = np.diag([1, 2, 3.0])
cov_pd = pd.DataFrame(cov_np)

test_cases = list(
    itertools.product(
        [_sim_pd, _sim_np, _sim_dict_pd, _sim_dict_np],  # simulate_moments
        [cov_np, cov_pd],  # moments_cov
        [{"algorithm": "scipy_lbfgsb"}, "scipy_lbfgsb"],  # optimize_options
    )
)


@pytest.mark.parametrize("simulate_moments, moments_cov, optimize_options", test_cases)
def test_estimate_msm(simulate_moments, moments_cov, optimize_options):
    start_params = np.array([3, 2, 1])

    expected_params = np.zeros(3)

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = simulate_moments(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    calculated = estimate_msm(
        simulate_moments=simulate_moments,
        empirical_moments=empirical_moments,
        moments_cov=moments_cov,
        params=start_params,
        optimize_options=optimize_options,
    )

    # check that minimization works
    aaae(calculated.params, expected_params)

    # assert that optimization result exists and is of correct type
    assert isinstance(calculated.optimize_result, OptimizeResult)

    # check that cov works
    calculated_cov = calculated.cov()
    if isinstance(calculated_cov, pd.DataFrame):
        calculated_cov = calculated_cov.to_numpy()

    # this works only in the very special case with diagonal moments cov and
    # jac = identity matrix
    expected_cov = np.diag([1, 2, 3])
    aaae(calculated_cov, expected_cov)
    aaae(calculated.se(), np.sqrt([1, 2, 3]))

    # works only because parameter point estimates are exactly zero
    aaae(calculated.p_values(), np.ones(3))

    expected_ci_upper = np.array([1.95996398, 2.77180765, 3.3947572])
    expected_ci_lower = -expected_ci_upper

    lower, upper = calculated.ci()
    aaae(lower, expected_ci_lower)
    aaae(upper, expected_ci_upper)

    aaae(calculated.ci(), calculated._ci)
    aaae(calculated.p_values(), calculated._p_values)
    aaae(calculated.se(), calculated._se)
    aaae(calculated.cov(), calculated._cov)

    summary = calculated.summary()
    aaae(summary["value"], np.zeros(3))
    aaae(summary["p_value"], np.ones(3))
    assert summary["stars"].tolist() == [""] * 3


def test_check_and_process_optimize_options_with_invalid_entries():
    with pytest.raises(ValueError):
        check_optimization_options({"criterion": lambda x: x}, "estimate_msm")


ls_test_cases = list(
    itertools.product(
        [_sim_pd, _sim_np, _sim_dict_pd, _sim_dict_np],  # simulate_moments
        [cov_np, cov_pd],  # moments_cov
        [{"algorithm": "pounders"}, "pounders"],  # optimize_options
    )
)


@pytest.mark.parametrize(
    "simulate_moments, moments_cov, optimize_options", ls_test_cases
)
def test_estimate_msm_ls(simulate_moments, moments_cov, optimize_options):
    start_params = np.array([3, 2, 1])

    expected_params = np.zeros(3)

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = simulate_moments(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    calculated = estimate_msm(
        simulate_moments=simulate_moments,
        empirical_moments=empirical_moments,
        moments_cov=moments_cov,
        params=start_params,
        optimize_options=optimize_options,
    )

    aaae(calculated.params, expected_params)


def test_estimate_msm_with_jacobian():
    start_params = np.array([3, 2, 1])

    expected_params = np.zeros(3)

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = _sim_np(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    calculated = estimate_msm(
        simulate_moments=_sim_np,
        empirical_moments=empirical_moments,
        moments_cov=cov_np,
        params=start_params,
        optimize_options="scipy_lbfgsb",
        jacobian=lambda x: np.eye(len(x)),
    )

    aaae(calculated.params, expected_params)
    aaae(calculated.cov(), cov_np)


def test_estimate_msm_with_algorithm_type():
    start_params = np.array([3, 2, 1])
    expected_params = np.zeros(3)
    empirical_moments = _sim_np(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    estimate_msm(
        simulate_moments=_sim_np,
        empirical_moments=empirical_moments,
        moments_cov=cov_np,
        params=start_params,
        optimize_options=scipy_optimizers.ScipyLBFGSB,
        jacobian=lambda x: np.eye(len(x)),
    )


def test_estimate_msm_with_algorithm():
    start_params = np.array([3, 2, 1])
    expected_params = np.zeros(3)
    empirical_moments = _sim_np(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    estimate_msm(
        simulate_moments=_sim_np,
        empirical_moments=empirical_moments,
        moments_cov=cov_np,
        params=start_params,
        optimize_options=scipy_optimizers.ScipyLBFGSB(stopping_maxfun=10),
        jacobian=lambda x: np.eye(len(x)),
    )


def test_to_pickle(tmp_path):
    start_params = np.array([3, 2, 1])

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = _sim_np(np.zeros(3))
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    calculated = estimate_msm(
        simulate_moments=_sim_np,
        empirical_moments=empirical_moments,
        moments_cov=cov_np,
        params=start_params,
        optimize_options="scipy_lbfgsb",
    )

    calculated.to_pickle(tmp_path / "bla.pkl")


def test_caching():
    start_params = np.array([3, 2, 1])

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = _sim_np(np.zeros(3))
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    got = estimate_msm(
        simulate_moments=_sim_np,
        empirical_moments=empirical_moments,
        moments_cov=cov_np,
        params=start_params,
        optimize_options="scipy_lbfgsb",
    )

    assert got._cache == {}
    cov = got.cov(method="robust", return_type="array")
    assert got._cache == {}
    cov = got.cov(method="robust", return_type="array", seed=0)
    assert_array_equal(list(got._cache.values())[0], cov)
