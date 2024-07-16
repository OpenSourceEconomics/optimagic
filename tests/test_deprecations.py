"""Test that our deprecations work.

This also serves as an internal overview of deprecated functions.

"""

import pytest
import numpy as np

from estimagic import minimize
from estimagic import maximize
from estimagic import first_derivative
from estimagic import second_derivative
from estimagic import get_benchmark_problems
from estimagic import run_benchmark
from estimagic import convergence_report
from estimagic import rank_report
from estimagic import traceback_report
from estimagic import profile_plot
from estimagic import convergence_plot
from estimagic import slice_plot
from estimagic import check_constraints
from estimagic import count_free_params
from estimagic import utilities
from estimagic import OptimizeLogReader, OptimizeResult
from estimagic import criterion_plot, params_plot
import optimagic as om

# ======================================================================================
# Deprecated in 0.5.0, remove in 0.6.0
# ======================================================================================


def test_estimagic_minimize_is_deprecated():
    with pytest.warns(FutureWarning, match="estimagic.minimize has been deprecated"):
        minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")


def test_estimagic_maximize_is_deprecated():
    with pytest.warns(FutureWarning, match="estimagic.maximize has been deprecated"):
        maximize(lambda x: -x @ x, np.arange(3), algorithm="scipy_lbfgsb")


def test_estimagic_first_derivative_is_deprecated():
    msg = "estimagic.first_derivative has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        first_derivative(lambda x: x @ x, np.arange(3))


def test_estimagic_second_derivative_is_deprecated():
    msg = "estimagic.second_derivative has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        second_derivative(lambda x: x @ x, np.arange(3))


def test_estimagic_benchmarking_functions_are_deprecated():
    msg = "estimagic.get_benchmark_problems has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        problems = get_benchmark_problems("example")

    msg = "estimagic.run_benchmark has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        results = run_benchmark(
            problems, optimize_options={"test": {"algorithm": "scipy_lbfgsb"}}
        )

    msg = "estimagic.convergence_report has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        convergence_report(problems, results)

    msg = "estimagic.rank_report has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        rank_report(problems, results)

    msg = "estimagic.traceback_report has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        traceback_report(problems, results)

    msg = "estimagic.profile_plot has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        profile_plot(problems, results)

    msg = "estimagic.convergence_plot has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        convergence_plot(problems, results)


def test_estimagic_slice_plot_is_deprecated():
    msg = "estimagic.slice_plot has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        slice_plot(
            func=lambda x: x @ x,
            params=np.arange(3),
            lower_bounds=np.zeros(3),
            upper_bounds=np.ones(3) * 5,
        )


def test_estimagic_check_constraints_is_deprecated():
    msg = "estimagic.check_constraints has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        check_constraints(
            params=np.arange(3), constraints=[{"loc": 0, "type": "fixed"}]
        )


def test_estimagic_count_free_params_is_deprecated():
    msg = "estimagic.count_free_params has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        count_free_params(
            params=np.arange(3), constraints=[{"loc": 0, "type": "fixed"}]
        )


@pytest.fixture()
def example_db(tmp_path):
    path = tmp_path / "test.db"

    def _crit(params):
        x = np.array(list(params.values()))
        return x @ x

    om.minimize(
        criterion=_crit,
        params={"a": 1, "b": 2, "c": 3},
        algorithm="scipy_lbfgsb",
        logging=path,
    )
    return path


def test_estimagic_log_reader_is_deprecated(example_db):
    msg = "estimagic.OptimizeLogReader has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        OptimizeLogReader(example_db)


def test_estimagic_optimize_result_is_deprecated():
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")

    msg = "estimagic.OptimizeResult has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        OptimizeResult(
            params=res.params,
            criterion=res.criterion,
            start_criterion=res.start_criterion,
            start_params=res.start_params,
            algorithm=res.algorithm,
            direction=res.direction,
            n_free=res.n_free,
        )


def test_estimagic_chol_params_to_lower_triangular_matrix_is_deprecated():
    msg = "estimagic.utilities.chol_params_to_lower_triangular_matrix has been deprecat"
    with pytest.warns(FutureWarning, match=msg):
        utilities.chol_params_to_lower_triangular_matrix(np.arange(6))


def test_estimagic_cov_params_to_matrix_is_deprecated():
    msg = "estimagic.utilities.cov_params_to_matrix has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_params_to_matrix(np.arange(6))


def test_estimagic_cov_matrix_to_params_is_deprecated():
    msg = "estimagic.utilities.cov_matrix_to_params has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_matrix_to_params(np.eye(3))


def test_estimagic_sdcorr_params_to_sds_and_corr_is_deprecated():
    msg = "estimagic.utilities.sdcorr_params_to_sds_and_corr has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.sdcorr_params_to_sds_and_corr(np.arange(6))


def test_estimagic_sds_and_corr_to_cov_is_deprecated():
    msg = "estimagic.utilities.sds_and_corr_to_cov has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.sds_and_corr_to_cov(np.arange(3), np.eye(3))


def test_estimagic_cov_to_sds_and_corr_is_deprecated():
    msg = "estimagic.utilities.cov_to_sds_and_corr has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_to_sds_and_corr(np.eye(3))


def test_estimagic_sdcorr_params_to_matrix_is_deprecated():
    msg = "estimagic.utilities.sdcorr_params_to_matrix has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.sdcorr_params_to_matrix(np.arange(6))


def test_estimagic_cov_matrix_to_sdcorr_params_is_deprecated():
    msg = "estimagic.utilities.cov_matrix_to_sdcorr_params has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_matrix_to_sdcorr_params(np.eye(3))


def test_estimagic_number_of_triangular_elements_to_dimension_is_deprecated():
    msg = "estimagic.utilities.number_of_triangular_elements_to_dimension has been"
    with pytest.warns(FutureWarning, match=msg):
        utilities.number_of_triangular_elements_to_dimension(6)


def test_estimagic_dimension_to_number_of_triangular_elements_is_deprecated():
    msg = "estimagic.utilities.dimension_to_number_of_triangular_elements has been"
    with pytest.warns(FutureWarning, match=msg):
        utilities.dimension_to_number_of_triangular_elements(3)


def test_estimagic_propose_alternatives_is_deprecated():
    msg = "estimagic.utilities.propose_alternatives has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.propose_alternatives("estimagic", list("abcdefg"))


def test_estimagic_robust_cholesky_is_deprecated():
    msg = "estimagic.utilities.robust_cholesky has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.robust_cholesky(np.eye(3))


def test_estimagic_robust_inverse_is_deprecated():
    msg = "estimagic.utilities.robust_inverse has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.robust_inverse(np.eye(3))


def test_estimagic_hash_array_is_deprecated():
    msg = "estimagic.utilities.hash_array has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.hash_array(np.arange(3))


def test_estimagic_calculate_trustregion_initial_radius_is_deprecated():
    msg = "estimagic.utilities.calculate_trustregion_initial_radius has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.calculate_trustregion_initial_radius(np.arange(3))


def test_estimagic_pickle_functions_are_deprecated(tmp_path):
    msg = "estimagic.utilities.to_pickle has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.to_pickle(np.arange(3), tmp_path / "test.pkl")

    msg = "estimagic.utilities.read_pickle has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.read_pickle(tmp_path / "test.pkl")


def test_estimagic_isscalar_is_deprecated():
    msg = "estimagic.utilities.isscalar has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.isscalar(1)


def test_estimagic_get_rng_is_deprecated():
    msg = "estimagic.utilities.get_rng has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.get_rng(42)


def test_estimagic_criterion_plot_is_deprecated():
    msg = "estimagic.criterion_plot has been deprecated"
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")
    with pytest.warns(FutureWarning, match=msg):
        criterion_plot(res)


def test_estimagic_params_plot_is_deprecated():
    msg = "estimagic.params_plot has been deprecated"
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")
    with pytest.warns(FutureWarning, match=msg):
        params_plot(res)