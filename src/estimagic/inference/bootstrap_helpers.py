import pandas as pd


def check_inputs(
    data=None,
    weight_by=None,
    cluster_by=None,
    ci_method="percentile",
    ci_level=0.95,
    skipdata=False,
):
    """Check validity of inputs.

    Args:
        data (pd.DataFrame): Dataset.
        weight_by (str): Column name of variable with weights.
        cluster_by (str): Column name of variable to cluster by.
        ci_method (str): Method of choice for computing confidence intervals.
            The default is "percentile".
        ci_level (float): Confidence level for the calculation of confidence
            intervals. The default is 0.95.
        skipdata (bool): Whether to skip all checks on the data argument.

    """
    ci_method_list = ["percentile", "bc", "t", "normal", "basic"]

    if not skipdata:
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise TypeError("Data must be a pandas.DataFrame or pandas.Series.")
        elif (weight_by is not None) and (weight_by not in data.columns.tolist()):
            raise ValueError(
                "Input 'weight_by' must be None or a column name of 'data'."
            )
        elif (cluster_by is not None) and (cluster_by not in data.columns.tolist()):
            raise ValueError(
                "Input 'cluster_by' must be None or a column name of 'data'."
            )

    if ci_method not in ci_method_list:
        msg = (
            "ci_method must be 'percentile', 'bc', 't', 'basic' or 'normal', "
            f"'{ci_method}' was supplied"
        )
        raise ValueError(msg)
    if ci_level > 1 or ci_level < 0:
        raise ValueError("Input 'ci_level' must be in [0,1].")
