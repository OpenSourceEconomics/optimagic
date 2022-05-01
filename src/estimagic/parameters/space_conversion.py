from functools import partial
from typing import NamedTuple

import estimagic.parameters.kernel_transformations as kt
import numpy as np
from estimagic.parameters.process_constraints import process_constraints
from estimagic.parameters.tree_conversion import FlatParams


def get_space_converter(
    flat_params,
    flat_constraints,
):
    """Get functions to convert between in-/external space of params and derivatives.


    Args:
        params ()


    Returns:
        SpaceConverter
        FlatParams: NamedTuple of 1d numpy array with flat and internal params and
            bounds.

    """
    transformations, constr_info = process_constraints(
        constraints=flat_constraints,
        params_vec=flat_params.values,
        lower_bounds=flat_params.lower_bounds,
        upper_bounds=flat_params.upper_bounds,
        param_names=flat_params.names,
    )

    _params_to_internal = partial(
        reparametrize_to_internal,
        internal_free=constr_info["_internal_free"],
        transformations=transformations,
    )

    _params_from_internal = partial(
        reparametrize_from_internal,
        fixed_values=constr_info["_internal_fixed_value"],
        pre_replacements=constr_info["_pre_replacements"],
        transformations=transformations,
        post_replacements=constr_info["_post_replacements"],
    )

    _dim_internal = int(constr_info["_internal_free"].sum())

    _pre_replace_jac = pre_replace_jacobian(
        pre_replacements=constr_info["_pre_replacements"], dim_in=_dim_internal
    )

    _post_replace_jac = post_replace_jacobian(
        post_replacements=constr_info["_post_replacements"]
    )

    _derivative_to_internal = partial(
        convert_external_derivative_to_internal,
        fixed_values=constr_info["_internal_fixed_value"],
        pre_replacements=constr_info["_pre_replacements"],
        transformations=transformations,
        pre_replace_jac=_pre_replace_jac,
        post_replace_jac=_post_replace_jac,
    )

    _has_transforming_constraints = bool(transformations)

    converter = SpaceConverter(
        params_to_internal=_params_to_internal,
        params_from_internal=_params_from_internal,
        derivative_to_internal=_derivative_to_internal,
        has_transforming_constraints=_has_transforming_constraints,
    )

    internal_params = FlatParams(
        values=converter.params_to_internal(flat_params.values),
        lower_bounds=constr_info["_internal_lower"],
        upper_bounds=constr_info["_internal_upper"],
        names=None,
    )

    return converter, internal_params


class SpaceConverter(NamedTuple):
    params_to_internal: callable
    params_from_internal: callable
    derivative_to_internal: callable
    has_transforming_constraints: bool


def reparametrize_to_internal(
    external,
    internal_free,
    transformations,
):
    """Convert a params DataFrame into a numpy array of internal parameters.

    Args:
        external (np.ndarray or pandas.DataFrmae): 1d array with of external parameter
            values or params DataFrame.
        internal_free (np.ndarray): 1d array of lenth n_external that determines
            which parameters are free.
        transformations (list): Processed transforming constraints.

    Returns:
        internal_params (numpy.ndarray): 1d numpy array of free reparametrized
            parameters.

    """
    with_internal_values = external.copy()

    for constr in transformations:
        func = getattr(kt, f"{constr['type']}_to_internal")

        with_internal_values[constr["index"]] = func(external[constr["index"]], constr)

    internal = with_internal_values[internal_free]

    return internal


def reparametrize_from_internal(
    internal,
    fixed_values,
    pre_replacements,
    transformations,
    post_replacements,
):
    """Convert a numpy array of internal parameters to a params DataFrame.

    Args:
        internal (numpy.ndarray): 1d numpy array with internal parameters
        fixed_values (numpy.ndarray): 1d numpy array of length n_external. It contains
            NaN for parameters that are not fixed and an internal representation of the
            value to which a parameter has been fixed for all others.
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        transformations (list): Processed transforming constraints.
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.

    Returns:
        numpy.ndarray: Array with external parameters

    """
    # do pre-replacements
    external_values = pre_replace(internal, fixed_values, pre_replacements)

    # do transformations
    for constr in transformations:
        func = getattr(kt, f"{constr['type']}_from_internal")
        external_values[constr["index"]] = func(
            external_values[constr["index"]], constr
        )

    # do post-replacements
    external_values = post_replace(external_values, post_replacements)

    return external_values


def convert_external_derivative_to_internal(
    external_derivative,
    internal_values,
    fixed_values,
    pre_replacements,
    transformations,
    post_replacements=None,
    pre_replace_jac=None,
    post_replace_jac=None,
):
    r"""Compute the derivative of the criterion utilizing an external derivative.

    Denote by :math:`c` the criterion function which is evaluated on the full
    parameter set. Denote by :math:`g` the paramater transform which maps an
    internal to an external paramter, i.e :math:`g: x \mapsto g(x)`, with
    :math:`x` denoting the internal paramter vector and :math:`g(x)` the
    respective external parameter frame. We are interested in the derivative of
    the composition :math:`f := c \circ g` which maps an internal vector to the
    criterion value. The derivative can be computed using the chain rule, as

    .. math::
        \frac{\mathrm{d}f}{\mathrm{d}x}(x) =
            \frac{\mathrm{d}c}{\mathrm{d}g}(g(x)) \times
            \frac{\mathrm{d}g}{\mathrm{d}x}(x)

    We assume that the user provides the first part of the above product. The
    second part denotes the derivative of the parameter transform from inner
    to external.

    Args:
        external_derivative (numpy.ndarray): The external derivative evaluated at
            external values mapped from ``internal_values``.
        internal_values (numpy.ndarray): 1d numpy array with internal parameters
        fixed_values (numpy.ndarray): 1d numpy array of length n_external. It contains
            NaN for parameters that are not fixed and an internal representation of the
            value to which a parameter has been fixed for all others.
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        transformations (list): Processed transforming constraints.
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.
        pre_replace_jac (np.ndarray): 2d Array with the jacobian of pre_replace
        post_replacment_jacobian (np.ndarray): 2d Array with the jacobian post_replace

    Returns:
        deriv (numpy.ndarray): The gradient or Jacobian.

    """
    dim_in = len(internal_values)

    pre_replaced = pre_replace(internal_values, fixed_values, pre_replacements)

    if post_replacements is None and post_replace_jac is None:
        raise ValueError(
            "either post_replacements or post_replace_jac must be specified."
        )

    if pre_replace_jac is None:
        pre_replace_jac = pre_replace_jacobian(pre_replacements, dim_in)

    if post_replace_jac is None:
        post_replace_jac = post_replace_jacobian(post_replacements)

    transform_jac = transformation_jacobian(transformations, pre_replaced)

    external_derivative = np.atleast_2d(external_derivative)
    tall_external = external_derivative.shape[0] > external_derivative.shape[1]

    mat_list = [
        external_derivative,
        post_replace_jac,
        transform_jac,
        pre_replace_jac,
    ]

    if tall_external:
        deriv = _multiply_from_right(mat_list)
    else:
        deriv = _multiply_from_left(mat_list)

    # return gradient with shape (len(params),)
    if deriv.shape[0] == 1:
        deriv = deriv.flatten()
    return deriv


def _multiply_from_left(mat_list):
    """Multiply all matrices in the list, starting from the left.

    Note that this only affects the order in which the pairwise multiplications happen,
    not the actual result.

    """
    out = mat_list[0]
    for mat in mat_list[1:]:
        out = out @ mat
    return out


def _multiply_from_right(mat_list):
    """Multiply all matrices in the list, starting from the right.

    Note that this only affects the order in which the pairwise multiplications happen,
    not the actual result.

    """
    out = mat_list[-1]
    for mat in reversed(mat_list[:-1]):
        out = mat @ out
    return out


def pre_replace(internal_values, fixed_values, pre_replacements):
    """Return pre-replaced parameters.

    Args:
        internal (numpy.ndarray): 1d numpy array with internal parameter.
        fixed_values (numpy.ndarray): 1d numpy array of length n_external. It contains
            NaN for parameters that are not fixed and an internal representation of the
            value to which a parameter has been fixed for all others.
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.

    Returns:
        pre_replaced (numpy.ndarray): 1d numpy array with pre-replaced params.


    Examples:

        >>> internal_values = np.array([1., 2.])
        >>> fixed_values = np.array([np.nan, 0, np.nan])
        >>> pre_replacements = np.array([1, -1, 0])
        >>> pre_replace(internal_values, fixed_values, pre_replacements)
        array([2., 0., 1.])

    """
    pre_replaced = fixed_values.copy()

    mask = pre_replacements >= 0
    positions = pre_replacements[mask]
    pre_replaced[mask] = internal_values[positions]
    return pre_replaced


def pre_replace_jacobian(pre_replacements, dim_in):
    """Return Jacobian of pre-replacement step.

    Remark. The function ``pre_replace`` can have ``np.nan`` in its output. In
    this case we know from the underlying structure that the derivative of this
    output with respect to any of the inputs is zero. Here we use this additional
    knowledge; however, when the derivative is computed using a numerical
    differentiation technique this will not be the case. Thus the numerical
    derivative can differ from the derivative here in these cases.

    Args:
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        dim_in (int): Dimension of the internal parameters.

    Returns:
        jacobian (np.ndarray): The jacobian.

    Examples:
        >>> # Note: The example is the same as in the doctest of pre_replace
        >>> pre_replacements = np.array([1, -1, 0])
        >>> pre_replace_jacobian(pre_replacements, 2)
        array([[0., 1.],
               [0., 0.],
               [1., 0.]])

    """
    dim_out = len(pre_replacements)
    mask = pre_replacements >= 0
    position_in = pre_replacements[mask]
    position_out = np.arange(dim_out)[mask]

    jacobian = np.zeros((dim_out, dim_in))
    jacobian[position_out, position_in] = 1
    return jacobian


def transformation_jacobian(transformations, pre_replaced):
    """Return Jacobian of constraint transformation step.

    The Jacobian of the constraint transformation step is build as a block matrix
    of either identity matrices, in the case when the external parameter equals
    the internal parameter, or, of the Jacobians of the specific kernel transforms,
    in case the external paramater is a transformed version of the internal.

    Args:
        transformations (list): Processed transforming constraints.
        pre_replaced (numpy.ndarray): 1d numpy array with pre-replaced params.
        dim (int): The dimension of the external parameters.

    Returns:
        jacobian (numpy.ndarray): The Jacobian.

    """
    dim = len(pre_replaced)
    jacobian = np.eye(dim)

    for constr in transformations:
        block_indices = constr["index"]
        jacobian_func = getattr(kt, f"{constr['type']}_from_internal_jacobian")
        jac = jacobian_func(pre_replaced[block_indices], constr)
        jacobian[np.ix_(block_indices, block_indices)] = jac

    return jacobian


def post_replace(external_values, post_replacements):
    """Return post-replaed parameters.

    Args:
        external_values (numpy.ndarray): 1d numpy array of external params.
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.

    Returns:
        post_replaced (numpy.ndarray): 1d numpy array with post-replaced params.

    Examples:
        >>> external_values = np.array([3., 4., np.nan])
        >>> post_replacements = np.array([-1, -1, 1])
        >>> post_replace(external_values, post_replacements)
        array([3., 4., 4.])

    """
    post_replaced = external_values.copy()

    mask = post_replacements >= 0
    positions = post_replacements[mask]
    post_replaced[mask] = post_replaced[positions]
    return post_replaced


def post_replace_jacobian(post_replacements):
    """Return Jacobian of post-replacement step.

    Args:
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.
        dim (int): The dimension of the external parameters.

    Returns:
        jacobian (np.ndarray): The Jacobian.

    Examples:
        >>> # Note: the example is the same as in the doctest of post_replace
        >>> post_replacements = np.array([-1, -1, 1])
        >>> post_replace_jacobian(post_replacements)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.]])



    """
    dim = len(post_replacements)
    mask = post_replacements >= 0
    positions_in = post_replacements[mask]
    positions_out = np.arange(dim)[mask]

    jacobian = np.eye(dim)
    jacobian[positions_out, :] *= 0
    jacobian[positions_out, positions_in] = 1
    return jacobian
