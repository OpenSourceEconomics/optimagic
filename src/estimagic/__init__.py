from estimagic import utilities
from estimagic.benchmarking.benchmarking import get_benchmark_problems
from estimagic.benchmarking.benchmarking import run_benchmark
from estimagic.differentiation.derivatives import first_derivative
from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.estimation.estimate_msm import estimate_msm
from estimagic.estimation.msm_weighting import get_moments_cov
from estimagic.inference.bootstrap import bootstrap
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize
from estimagic.visualization.convergence_plot import convergence_plot
from estimagic.visualization.profile_plot import profile_plot

try:
    from ._version import version as __version__
except ImportError:
    # broken installation, we don't even try unknown only works because we do poor mans
    # version compare
    __version__ = "unknown"


__all__ = [
    "maximize",
    "minimize",
    "utilities",
    "first_derivative",
    "bootstrap",
    "estimate_msm",
    "estimate_ml",
    "get_moments_cov",
    "get_benchmark_problems",
    "run_benchmark",
    "profile_plot",
    "convergence_plot",
    "__version__",
]
