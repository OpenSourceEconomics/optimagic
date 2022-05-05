"""Aggregate the multiple parameter and function output conversions into on."""
from typing import NamedTuple

from estimagic.parameters.process_selectors import process_selectors
from estimagic.parameters.scale_conversion import get_scale_converter
from estimagic.parameters.space_conversion import get_space_converter
from estimagic.parameters.tree_conversion import get_tree_converter


def get_converter(
    func,
    params,
    constraints,
    lower_bounds,
    upper_bounds,
    func_eval,
    primary_key,
    scaling,
    scaling_options,
    derivative_eval=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    add_soft_bounds=False,
):
    """Get a converter between external and internal params and internal params.



    Returns:
        Converter: NamedTuple with methods to convert between internal and external
            parameters, derivatives and function outputs.
        FlatParams: NamedTuple with internal parameter values, lower_bounds and
            upper_bounds.

    """
    tree_converter, flat_params = get_tree_converter(
        params=params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        func_eval=func_eval,
        derivative_eval=derivative_eval,
        primary_key=primary_key,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        add_soft_bounds=add_soft_bounds,
    )

    flat_constraints = process_selectors(
        constraints=constraints,
        params=params,
        tree_converter=tree_converter,
    )

    space_converter, internal_params = get_space_converter(
        flat_params=flat_params, flat_constraints=flat_constraints
    )

    def _helper(x):
        x_external = space_converter.params_from_internal(x)
        x_tree = tree_converter.params_unflatten(x_external)
        f_raw = func(x_tree)
        f_flat = tree_converter.func_flatten(f_raw)
        f_agg = aggregate_func_output_to_value(f_flat, primary_key)
        return f_agg

    scale_converter, scaled_params = get_scale_converter(
        flat_params=internal_params,
        func=_helper,
        scaling=scaling,
        scaling_options=scaling_options,
    )

    def _params_to_internal(params):
        x_flat = tree_converter.params_flatten(params)
        x_internal = space_converter.params_to_internal(x_flat)
        x_scaled = scale_converter.params_to_internal(x_internal)
        return x_scaled

    def _params_from_internal(x, return_type="tree"):
        x_unscaled = scale_converter.params_from_internal(x)
        x_external = space_converter.params_from_internal(x_unscaled)

        x_tree = tree_converter.params_unflatten(x_external)
        if return_type == "tree":
            out = x_tree
        elif return_type == "tree_and_flat":
            out = x_tree, x_external
        elif return_type == "flat":
            out = x_external
        else:
            raise ValueError()  # xxxx
        return out

    def _derivative_to_internal(derivative_eval, x):
        jacobian = tree_converter.derivative_flatten(derivative_eval)
        x_unscaled = scale_converter.params_from_internal(x)
        jac_with_space_conversion = space_converter.derivative_to_internal(
            jacobian, x_unscaled
        )
        jac_with_unscaling = scale_converter.derivative_to_internal(
            jac_with_space_conversion
        )
        return jac_with_unscaling

    def _func_to_internal(func_eval):
        return tree_converter.func_flatten(func_eval)

    flat_params = scaled_params._replace(free_mask=internal_params.free_mask)

    converter = Converter(
        params_to_internal=_params_to_internal,
        params_from_internal=_params_from_internal,
        derivative_to_internal=_derivative_to_internal,
        func_to_internal=_func_to_internal,
        has_transforming_constraints=space_converter.has_transforming_constraints,
    )

    return converter, flat_params


class Converter(NamedTuple):
    params_to_internal: callable
    params_from_internal: callable
    derivative_to_internal: callable
    func_to_internal: callable
    has_transforming_constraints: bool


def aggregate_func_output_to_value(f_eval, primary_key):
    if primary_key == "value":
        return f_eval
    elif primary_key == "contributions":
        return f_eval.sum()
    elif primary_key == "root_contributions":
        return f_eval @ f_eval