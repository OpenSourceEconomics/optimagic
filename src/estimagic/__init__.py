from estimagic import utilities
from estimagic.differentiation.derivatives import first_derivative
from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.estimation.estimate_msm import estimate_msm
from estimagic.estimation.msm_weighting import get_moments_cov
from estimagic.inference.bootstrap import bootstrap
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize

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
    "__version__",
]