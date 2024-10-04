from pathlib import Path

import pandas as pd
import plotly.express as px
from packaging import version

DOCS_DIR = Path(__file__).parent.parent / "docs"
OPTIMAGIC_ROOT = Path(__file__).parent

PLOTLY_TEMPLATE = "simple_white"
PLOTLY_PALETTE = px.colors.qualitative.Set2

DEFAULT_N_CORES = 1

CRITERION_PENALTY_SLOPE = 0.1
CRITERION_PENALTY_CONSTANT = 100

# ======================================================================================
# Check Available Packages
# ======================================================================================

try:
    from petsc4py import PETSc  # noqa: F401
except ImportError:
    IS_PETSC4PY_INSTALLED = False
else:
    IS_PETSC4PY_INSTALLED = True

try:
    import nlopt  # noqa: F401
except ImportError:
    IS_NLOPT_INSTALLED = False
else:
    IS_NLOPT_INSTALLED = True

try:
    import pybobyqa  # noqa: F401
except ImportError:
    IS_PYBOBYQA_INSTALLED = False
else:
    IS_PYBOBYQA_INSTALLED = True

try:
    import dfols  # noqa: F401
except ImportError:
    IS_DFOLS_INSTALLED = False
else:
    IS_DFOLS_INSTALLED = True

try:
    import pygmo  # noqa: F401
except ImportError:
    IS_PYGMO_INSTALLED = False
else:
    IS_PYGMO_INSTALLED = True

try:
    import cyipopt  # noqa: F401
except ImportError:
    IS_CYIPOPT_INSTALLED = False
else:
    IS_CYIPOPT_INSTALLED = True

try:
    import fides  # noqa: F401
except ImportError:
    IS_FIDES_INSTALLED = False
else:
    IS_FIDES_INSTALLED = True

try:
    import jax  # noqa: F401
except ImportError:
    IS_JAX_INSTALLED = False
else:
    IS_JAX_INSTALLED = True


try:
    import tranquilo  # noqa: F401
except ImportError:
    IS_TRANQUILO_INSTALLED = False
else:
    IS_TRANQUILO_INSTALLED = True


try:
    import numba  # noqa: F401
except ImportError:
    IS_NUMBA_INSTALLED = False
else:
    IS_NUMBA_INSTALLED = True


# ======================================================================================
# Check if pandas version is newer or equal to version 2.1.0
# ======================================================================================

IS_PANDAS_VERSION_NEWER_OR_EQUAL_TO_2_1_0 = version.parse(
    pd.__version__
) >= version.parse("2.1.0")
