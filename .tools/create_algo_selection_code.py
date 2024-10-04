import importlib
import inspect
import pkgutil
import textwrap
from itertools import combinations
from types import ModuleType
from typing import Callable, Type

from optimagic.config import OPTIMAGIC_ROOT
from optimagic.optimization.algorithm import Algorithm
from optimagic.typing import AggregationLevel


def main():
    """Create the source code for algorithms.py.

    The main part are nested dataclasses that enable filtered autocomplete for algorithm
    selection.

    """
    # create some basic inputs
    docstring = _get_docstring_code()
    modules = _import_optimizer_modules("optimagic.optimizers")
    all_algos = _get_all_algorithms(modules)
    filters = _get_filters()
    all_categories = list(filters)
    selection_info = _create_selection_info(all_algos, all_categories)

    # create the code for imports
    imports = _get_imports(modules)

    # create the code for the ABC AlgoSelection class
    parent_class_snippet = _get_base_class_code()

    # create the code for the dataclasses
    dataclass_snippets = []
    for active_categories in selection_info:
        new_snippet = create_dataclass_code(
            active_categories=active_categories,
            all_categories=all_categories,
            selection_info=selection_info,
        )
        dataclass_snippets.append(new_snippet)

    # create the code for the instantiation
    instantiation_snippet = _get_instantiation_code()

    # write code to the file

    with open(OPTIMAGIC_ROOT / "algorithms.py", "w") as f:
        f.write(docstring)
        f.write(imports + "\n\n")
        f.write(parent_class_snippet + "\n")
        f.write("\n\n".join(dataclass_snippets))
        f.write("\n\n")
        f.write(instantiation_snippet)


# ======================================================================================
# Functions to collect algorithms
# ======================================================================================


def _import_optimizer_modules(package_name):
    """Collect all public modules in a given package in a list."""
    package = importlib.import_module(package_name)
    modules = []

    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module_parts = module_name.split(".")
        if all(not part.startswith("_") for part in module_parts) and not is_pkg:
            module = importlib.import_module(module_name)
            modules.append(module)

    return modules


def _get_all_algorithms(modules: list[ModuleType]) -> dict[str, Type[Algorithm]]:
    """Collect all algorithms in moudles."""
    out = {}
    for module in modules:
        out.update(_get_algorithms_in_module(module))
    return out


def _get_algorithms_in_module(module: ModuleType) -> dict[str, Type[Algorithm]]:
    """Collect all algorithms in a single module."""
    candidate_dict = dict(inspect.getmembers(module, inspect.isclass))
    candidate_dict = {
        k: v for k, v in candidate_dict.items() if hasattr(v, "__algo_info__")
    }
    algos = {}
    for candidate in candidate_dict.values():
        name = candidate.__algo_info__.name
        if issubclass(candidate, Algorithm) and candidate is not Algorithm:
            algos[name] = candidate
    return algos


# ======================================================================================
# Functions to filter algorithms by selectors
# ======================================================================================
def _is_gradient_based(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.needs_jac


def _is_gradient_free(algo: Type[Algorithm]) -> bool:
    return not _is_gradient_based(algo)


def _is_global(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.is_global


def _is_local(algo: Type[Algorithm]) -> bool:
    return not _is_global(algo)


def _is_bounded(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.supports_bounds


def _is_linear_constrained(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.supports_linear_constraints


def _is_nonlinear_constrained(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.supports_nonlinear_constraints


def _is_scalar(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.solver_type == AggregationLevel.SCALAR


def _is_least_squares(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.solver_type == AggregationLevel.LEAST_SQUARES


def _is_likelihood(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.solver_type == AggregationLevel.LIKELIHOOD


def _is_parallel(algo: Type[Algorithm]) -> bool:
    return algo.__algo_info__.supports_parallelism


def _get_filters() -> dict[str, Callable[[Type[Algorithm]], bool]]:
    """Create a dict mapping from category names to filter functions."""
    filters = {
        "GradientBased": _is_gradient_based,
        "GradientFree": _is_gradient_free,
        "Global": _is_global,
        "Local": _is_local,
        "Bounded": _is_bounded,
        "LinearConstrained": _is_linear_constrained,
        "NonlinearConstrained": _is_nonlinear_constrained,
        "Scalar": _is_scalar,
        "LeastSquares": _is_least_squares,
        "Likelihood": _is_likelihood,
        "Parallel": _is_parallel,
    }
    return filters


# ======================================================================================
# Functions to create a mapping from a tuple of selectors to subsets of the dict
# mapping algorithm names to algorithm classes
# ======================================================================================


def _create_selection_info(
    all_algos: dict[str, Type[Algorithm]],
    categories: list[str],
) -> dict[tuple[str, ...], dict[str, Type[Algorithm]]]:
    """Create a dict mapping from a tuple of selectors to subsets of the all_algos dict.

    Args:
        all_algos: Dictionary mapping algorithm names to algorithm classes.
        categories: List of categories to filter by.

    Returns:
        A dictionary mapping tuples of selectors to dictionaries of algorithm names
            and their corresponding classes.

    """
    category_combinations = _generate_category_combinations(categories)
    out = {}
    for comb in category_combinations:
        filtered_algos = _apply_filters(all_algos, comb)
        if filtered_algos:
            out[comb] = filtered_algos
    return out


def _generate_category_combinations(categories: list[str]) -> list[tuple[str, ...]]:
    """Generate all combinations of categories, sorted by length in descending order.

    Args:
        categories: A list of category names.

    Returns:
        A list of tuples, where each tuple represents a combination of categories.

    """
    result = []
    for r in range(len(categories) + 1):
        result.extend(map(tuple, map(sorted, combinations(categories, r))))
    return sorted(result, key=len, reverse=True)


def _apply_filters(
    all_algos: dict[str, Type[Algorithm]], categories: tuple[str, ...]
) -> dict[str, Type[Algorithm]]:
    """Apply filters to the algorithms based on the given categories.

    Args:
        all_algos: A dictionary mapping algorithm names to algorithm classes.
        categories: A tuple of category names to filter by.

    Returns:
        filtered dictionary of algorithms that match all given categories.

    """
    filtered = all_algos
    filters = _get_filters()
    for category in categories:
        filter_func = filters[category]
        filtered = {name: algo for name, algo in filtered.items() if filter_func(algo)}
    return filtered


# ======================================================================================
# Functions to create code for the dataclasses
# ======================================================================================


def create_dataclass_code(
    active_categories: tuple[str, ...],
    all_categories: list[str],
    selection_info: dict[tuple[str, ...], dict[str, Type[Algorithm]]],
):
    """Create the source code for a dataclass representing a selection of algorithms.

    Args:
        active_categories: A tuple of active category names.
        all_categories: A list of all category names.
        selection_info: A dictionary that maps tuples of category names to dictionaries
            of algorithm names and their corresponding classes.

    Returns:
        A string containing the source code for the dataclass.

    """
    # get the children of the active categories
    children = _get_children(active_categories, all_categories, selection_info)

    # get the name of the class to be generated
    class_name = _get_class_name(active_categories)

    # get code for the dataclass fields
    field_template = "    {name}: Type[{class_name}] = {class_name}"
    field_strings = []
    for name, algo_class in selection_info[active_categories].items():
        field_strings.append(
            field_template.format(name=name, class_name=algo_class.__name__)
        )
    fields = "\n".join(field_strings)

    # get code for the properties to select children
    child_template = textwrap.dedent("""
        @property
        def {new_category}(self) -> {class_name}:
            return {class_name}()
    """)
    child_template = textwrap.indent(child_template, "    ")
    child_strings = []
    for new_category, categories in children.items():
        child_class_name = _get_class_name(categories)
        child_strings.append(
            child_template.format(
                new_category=new_category, class_name=child_class_name
            )
        )
    children_code = "\n".join(child_strings)

    # assemble the class
    out = "@dataclass(frozen=True)\n"
    out += f"class {class_name}(AlgoSelection):\n"
    out += fields + "\n"
    if children:
        out += children_code

    return out


def _get_class_name(active_categories: tuple[str, ...]) -> str:
    """Get the name of the class based on the active categories."""
    return "".join(active_categories) + "Algorithms"


def _get_children(
    active_categories: tuple[str, ...],
    all_categories: list[str],
    selection_info: dict[tuple[str, ...], dict[str, Type[Algorithm]]],
) -> list[tuple[str, ...]]:
    """Get the children of the active categories.

    Args:
        active_categories: A tuple of active category names.
        all_categories: A list of all category names.
        selection_info: A dictionary that maps tuples of category names to dictionaries
            of algorithm names and their corresponding classes.

    Returns:
        A dict mapping additional categories to a sorted tuple of categories
            that contains all active categories and the additional category. Entries
            are only included if the selected categories are in `selection_info`, i.e.
            if there exist algorithms that are compatible with all categories.

    """
    inactive_categories = sorted(set(all_categories) - set(active_categories))
    out = {}
    for new_cat in inactive_categories:
        new_comb = tuple(sorted(active_categories + (new_cat,)))
        if new_comb in selection_info:
            out[new_cat] = new_comb
    return out


# ======================================================================================
# Functions to create the imports
# ======================================================================================


def _get_imports(modules: list[ModuleType]) -> str:
    """Create source code to import all algorithms."""
    snippets = [
        "from typing import Type",
        "from dataclasses import dataclass",
        "from optimagic.optimization.algorithm import Algorithm",
        "from typing import cast",
    ]
    for module in modules:
        algorithms = _get_algorithms_in_module(module)
        class_names = [algo.__name__ for algo in algorithms.values()]
        for class_name in class_names:
            snippets.append(f"from {module.__name__} import {class_name}")
    return "\n".join(snippets)


# ======================================================================================
# Functions to create the static parts of the code
# ======================================================================================


def _get_base_class_code() -> str:
    """Get the source code for the AlgoSelection class."""
    out = textwrap.dedent("""
        @dataclass(frozen=True)
        class AlgoSelection:

            @property
            def All(self) -> list[Type[Algorithm]]:
                raw = [field.default for field in self.__dataclass_fields__.values()]
                return cast(list[Type[Algorithm]], raw)

            @property
            def Available(self) -> list[Type[Algorithm]]:
                return [
                    a for a in self.All if a.__algo_info__.is_available # type: ignore
                ]

            @property
            def _all_algorithms_dict(self) -> dict[str, Type[Algorithm]]:
                return {a.__algo_info__.name: a for a in self.All} # type: ignore

            @property
            def _available_algorithms_dict(self) -> dict[str, Type[Algorithm]]:
                return {a.__algo_info__.name: a for a in self.Available} # type: ignore

    """)
    return out


def _get_docstring_code() -> str:
    """Get the source code for the docstring of the AlgoSelection class."""
    raw = (
        '"""This code was auto-generated by a pre-commit hook and should not be '
        "changed.\n\nIf you manually change this code, all of your changes will be "
        "overwritten the next time\nthe pre-commit hook runs.\n\nDetailed information "
        "on the purpose of the code can be found here:\n"
        "https://optimagic.readthedocs.io/en/latest/development/ep-02-typing.html#"
        'algorithm-selection\n\n"""\n'
    )
    out = textwrap.dedent(raw)
    return out


def _get_instantiation_code() -> str:
    """Get the source code for instantiating some classes at the end of the module."""
    out = textwrap.dedent("""
        algos = Algorithms()
        global_algos = GlobalAlgorithms()

        ALL_ALGORITHMS = algos._all_algorithms_dict
        AVAILABLE_ALGORITHMS = algos._available_algorithms_dict
        GLOBAL_ALGORITHMS = global_algos._available_algorithms_dict
    """)
    return out


if __name__ == "__main__":
    main()
