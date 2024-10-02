"""Check option dictionaries for minimize, maximize."""


def check_optimization_options(options, usage, algorithm_mandatory=True):
    """Check optimize_options or maximize_options for usage in estimation functions."""
    options = {} if options is None else options

    if algorithm_mandatory:
        if not isinstance(options, dict) or "algorithm" not in options:
            raise ValueError(
                "optimize_options or maximize_options must be a dict containing at "
                "least the entry 'algorithm'"
            )
    else:
        if not isinstance(options, dict):
            raise ValueError(
                "optimize_options or maximize_options must be a dict or None."
            )

    criterion_options = {
        "criterion",
        "criterion_kwargs",
        "derivative",
        "derivative_kwargs",
    }

    invalid_criterion = criterion_options.intersection(options)
    if invalid_criterion:
        msg = (
            "Entries related to the criterion function, its derivatives or keyword "
            "arguments of those functions are not valid entries of optimize_options "
            f"or maximize_options for {usage}. Remove: {invalid_criterion}"
        )
        raise ValueError(msg)

    general_options = {"logging", "log_options", "constraints"}

    invalid_general = general_options.intersection(options)

    if invalid_general:
        msg = (
            "The following are not valid entries of optimize_options because they are "
            "not only relevant for minimization but also for inference: "
            f"{invalid_general}"
        )
        raise ValueError(msg)
