import inspect
from types import ModuleType
from typing import Any

import optimagic as om
from optimagic import optimizers
from optimagic.algorithms import Algorithm


def _explore_module(module: ModuleType, prefix: str = "") -> dict[str, Any]:
    """Explore a module and return a dictionary of objects.

    For submodules, the prefix is the name of the parent module.

    Args:
        module: A module to explore.
        prefix: A prefix to add to the names of the objects.

    Returns:
        A dictionary of objects.

    """
    objects = {}
    for name, obj in inspect.getmembers(module):
        if not name.startswith("_"):  # Exclude private/special attributes
            full_name = f"{prefix}.{name}" if prefix else name
            objects[full_name] = obj
            if inspect.ismodule(obj) and obj.__name__.startswith(module.__name__):
                objects.update(_explore_module(obj, full_name))
    return objects


def _get_algorithm_classes(dict_of_objects: dict[str, Any]) -> dict[str, Any]:
    """Get all algorithm sub-classes from a dictionary of objects."""
    algorithms = {}
    for name, obj in dict_of_objects.items():
        try:
            is_algorithm = issubclass(obj, Algorithm) and obj is not Algorithm
        except TypeError:
            is_algorithm = False

        if is_algorithm:
            algorithms[name] = obj

    return algorithms


def test_algorithm_classes_match_om_algos_all():
    all_objects = _explore_module(optimizers)
    algorithm_classes = _get_algorithm_classes(all_objects)

    assert set(om.algos.All) == set(
        algorithm_classes.values()
    ), "Manually explored algorithm classes do not match om.algos.All"
