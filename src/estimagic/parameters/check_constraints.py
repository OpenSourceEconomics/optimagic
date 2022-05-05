"""Check compatibility of pc with each other and with bounds and fixes.

See the module docstring of process_constraints for naming conventions.
"""
import warnings

import numpy as np
import pandas as pd
from estimagic.exceptions import InvalidParamsError
from estimagic.utilities import cov_params_to_matrix
from estimagic.utilities import sdcorr_params_to_matrix


def check_constraints_are_satisfied(flat_constraints, param_values, param_names):
    """Check that params satisfies all constraints.

    This should be called before the more specialized constraints are rewritten to
    linear constraints in order to get better error messages!

    We let the checks pass if all "values" are np.nan. This way `process_constraints`
    can be used on empty params DataFrames which is useful to construct templates for
    start parameters that can be filled out by the user.

    Args:
        pc (list): List of constraints with processed selectors.
        params (pd.DataFrame): See :ref:`params`

    Raises:
        ValueError if constraints are not satisfied.

    """
    # skip check if all parameters are NaN
    if not np.isfinite(param_values).any():
        return

    for constr in flat_constraints:
        typ = constr["type"]
        subset = param_values[constr["index"]]

        base_msg = _get_message(constr, param_names)

        if typ == "covariance":
            cov = cov_params_to_matrix(subset)
            e, _ = np.linalg.eigh(cov)
            if not np.all(e > -1e-8):
                raise InvalidParamsError(base_msg.format(""))
        elif typ == "sdcorr":
            cov = sdcorr_params_to_matrix(subset)
            e, _ = np.linalg.eigh(cov)
            if not np.all(e > -1e-8):
                raise InvalidParamsError(base_msg.format(""))
        elif typ == "probability":
            if not np.isclose(subset.sum(), 1, rtol=0.01):
                explanation = "Probabilities do not sum to 1."
                raise InvalidParamsError(base_msg.format(explanation))
            if np.any(subset < 0):
                explanation = "There are negative Probabilities."
                raise InvalidParamsError(base_msg.format(explanation))
            if np.any(subset > 1):
                explanation = "There are probabilities larger than 1."
                raise InvalidParamsError(base_msg.format(explanation))
        elif typ == "increasing":
            if np.any(np.diff(subset) < 0):
                raise InvalidParamsError(base_msg.format(""))
        elif typ == "decreasing":
            if np.any(np.diff(subset) > 0):
                InvalidParamsError(base_msg.format(""))
        elif typ == "linear":
            wsum = subset.dot(constr["weights"])
            if "lower_bound" in constr and wsum < constr["lower_bound"]:
                explanation = "Lower bound of linear constraint is violated."
                raise InvalidParamsError(base_msg.format(explanation))
            elif "upper_bound" in constr and wsum > constr["upper_bound"]:
                explanation = "Upper bound of linear constraint violated"
                raise InvalidParamsError(base_msg.format(explanation))
            elif "value" in constr and not np.isclose(wsum, constr["value"]):
                explanation = "Equality condition of linear constraint violated"
                raise InvalidParamsError(base_msg.format(explanation))
        elif typ == "equality":
            if len(set(subset.tolist())) != 1:
                raise InvalidParamsError(base_msg.format(""))


def _get_message(constraint, param_names):
    start = f"{constraint['type']} is not fulfilled in params."

    explanation = "{}"

    names = [param_names[i] for i in constraint["index"]]

    end = (
        f"The names of the involved parameters is:\n{names}\n"
        "The relevant constraint (with processed selector fields) is:\n"
        f"{constraint}."
    )

    msg = start + explanation + end
    return msg


def check_types(constraints):
    """Check that no invalid constraint types are requested.

    Args:
        constraints (list): List of constraints.

    Raises:
        TypeError if invalid constraint types are encountered



    """
    valid_types = {
        "covariance",
        "sdcorr",
        "linear",
        "probability",
        "increasing",
        "decreasing",
        "equality",
        "pairwise_equality",
        "fixed",
    }
    for constr in constraints:
        if constr["type"] not in valid_types:
            raise TypeError("Invalid constraint_type: {}".format(constr["type"]))


def check_for_incompatible_overlaps(constr_info, transformations, parnames):
    """Check that there are no overlaps between constraints that transform parameters.

    Since the constraints are already consolidated such that only those that transform
    a parameter are left and all equality constraints are already plugged in, this
    boils down to checking that no parameter appears more than once.

    Args:
        constr_info (dict): Dict of 1d numpy arrays with info about constraints.
        transformations (list): Processed transforming constraints.
        parnames (list): List of parameter names.

    """
    all_indices = []
    for constr in transformations:
        all_indices += constr["index"]

    msg = (
        "Transforming constraints such as 'covariance', 'sdcorr', 'probability' "
        "and 'linear' cannot overlap. This includes overlaps induced by equality "
        "constraints. This was violated for the following parameters:\n{}"
    )

    if len(set(all_indices)) < len(all_indices):
        unique, counts = np.unique(all_indices, return_counts=True)
        invalid_indices = unique[counts >= 2]
        invalid_names = parnames[invalid_indices]

        raise ValueError(msg.format(invalid_names))


def check_fixes_and_bounds(constr_info, transformations, parnames):
    """Check fixes.

    Warn the user if he fixes a parameter to a value even though that parameter has
    a different non-nan value in params

    check that fixes are compatible with other constraints.

    Args:
        constr_info (dict): Dict of 1d numpy arrays with info about constraints.
        transformations (list): Processed transforming constraints.
        parnames (list): List of parameter names.

    """
    df = pd.DataFrame(constr_info, index=parnames)
    # warn about fixes to a different value that what is in the "value" column
    problematic_fixes = df.query(
        "value != _fixed_value & _fixed_value.notnull() & value.notnull()",
        engine="python",
    )

    warn_msg = (
        "The following parameters were fixed to a different value than their start "
        "value. The start values are overwritten with the fixed values. "
        "You can ignore this message if you did this on purpose. :\n\n {}."
    )

    if len(problematic_fixes) > 0:
        warnings.warn(warn_msg.format(problematic_fixes[["value", "_fixed_value"]]))

    # Check fixes and bounds are compatible with other constraints
    prob_msg = (
        "{} constraints are incompatible with fixes or bounds. "
        "This is violated for:\n{}"
    )

    cov_msg = (
        "{} constraints are incompatible with fixes or bounds except for the first "
        "parameter. This is violated for:\n{}"
    )

    for constr in transformations:
        if constr["type"] in ["covariance", "sdcorr"]:
            subset = df.iloc[constr["index"][1:]]
            if subset["_is_fixed_to_value"].any():
                problematic = subset[subset["_is_fixed_to_value"]].index
                raise ValueError(cov_msg.format(constr["type"], problematic))
            if np.isfinite(subset[["lower_bound", "upper_bound"]]).any(axis=None):
                problematic = (
                    subset.replace([-np.inf, np.inf], np.nan).dropna(how="all").index
                )
                raise ValueError(cov_msg.format(constr["type"], problematic))
        elif constr["type"] == "probability":
            subset = df.iloc[constr["index"]]
            if subset["_is_fixed_to_value"].any():
                problematic = subset[subset["_is_fixed_to_value"]].index
                raise ValueError(prob_msg.format(constr["type"], problematic))
            if np.isfinite(subset[["lower_bound", "upper_bound"]]).any(axis=None):
                problematic = (
                    subset.replace([-np.inf, np.inf], np.nan).dropna(how="all").index
                )
                raise ValueError(prob_msg.format(constr["type"], problematic))

    invalid = df.query("lower_bound >= upper_bound")[["lower_bound", "upper_bound"]]
    msg = (
        "lower_bound must be strictly smaller than upper_bound. "
        + f"This is violated for:\n{invalid}"
    )
    if len(invalid) > 0:
        raise ValueError(msg)
