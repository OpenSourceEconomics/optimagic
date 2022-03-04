import re
from collections import namedtuple
from copy import deepcopy
from functools import partial
from warnings import warn

import numpy as np
import pandas as pd


def estimation_table(
    models,
    *,
    return_type="dataframe",
    render_options=None,
    custom_param_names=None,
    show_col_names=True,
    show_col_groups=False,
    custom_col_names=None,
    custom_col_groups=None,
    show_index_names=False,
    custom_index_names=None,
    show_inference=True,
    confidence_intervals=False,
    show_stars=True,
    significance_levels=(0.1, 0.05, 0.01),
    number_format=("{0:.3g}", "{0:.5f}", "{0:.4g}"),
    add_trailing_zeros=True,
    padding=1,
    show_footer=True,
    stat_keys=None,
    append_notes=True,
    notes_label="Note:",
    custom_notes=None,
    siunitx_warning=True,
    alignment_warning=True,
):
    r"""Generate html and LaTex tables provided (lists of) of models.

    Can return strings of LaTex/html scripts or dictionaries with processed dataframes
    to be passed to tabular functions, or save tables to path.

    Args:
        models (list): list of estimation results. The models can come from
            statmodels or be constructed from the outputs of `estimagic.estimate_ml`
            or `estimagic.estimate_msm`. With a little bit of work it is also possible
            to construct them out of R or other results. If a model is not a
            statsmodels results they must be dictionaries or namedtuples with the
            following entries: "params" (a DataFrame with value column), "info"
            (a dictionary with summary statistics such as "n_obs", "rsquared", ...)
            and "name" (a string), or a DataFrame with value column.
            If a models is a statsmodels result, model.endog_names is used as name and
            the rest is extracted from corresponding statsmodels attributes. The model
            names do not have to be unique but if they are not, models with the same
            name dneed to be grouped together.
        return_type (str): Can be "dataframe", "latex", "html", "render_inputs" or a
            file path with the extension .tex or .html. If "render_inputs" is passed,
            a dictionary with the entries "body", "footer", "notes" and other
            information is returned. The entries can be modified by the user (
            e.g. formatting changes, renaming of columns index, ...) and then passed
            to `render_latex` or render_htm`.
        custom_param_names (dict): Dictionary that is used to rename parameters. The
            keys are the old parameter names or index entries. The values are
            the new names. Default None.
        show_index_names (bool): If True, the index names are displayed. Default False.
        custom_index_names (dict or list): Dictionary or list to set the names of the
            index levels of the parameters. Only used if "add_index_names" is set to
            True in the render_options. Default None.
        render_options (dict): a dictionary with keyword arguments that are passed to
            df.to_latex or df.to_html, depending on the return_type.
            The default is None.
        show_col_names (bool): If True, the column names are displayed. Default True.
        show_col_groups (bool): If True, the column groups are displayed. Default False.
        custom_col_names (dict or list): A list of column names or dict to rename the
            default column names. The default column names are the model names if the
            model names are unique, otherwise (1), (2), etc..
        custom_col_groups (dict or list): A list of column group or dict to rename
            the default column groups. The default column groups are the model names
            if the model names are not unique and undefined otherwise.
        inference_type (str or None): One of "standard_errors", "confidenece_intervals"
            or None. Determines which kind of inference measure is displayed in the
            table.
        show_stars (bool): a boolean variable for printing significance stars.
            Default is True.
        significance_levels (list): a list of floats for p value's significance cutt-off
            values. Default is [0.1,0.05,0.01].
        number_format (int): an integer for the number of digits to the right of the
            decimal point to round to. Default is 2.
        padding (int): an integer used for aligning LaTex columns. Affects the
            alignment of the columns to the left of the decimal point of numerical
            entries. Default is 1. If the number of models is more than 2, set the
            value of padding to 3 or more to avoid columns overlay in the tex output.
        show_footer (bool): a boolean variable for displaying statistics, e.g. R2,
            Obs numbers. Default is True.
        stat_keys (dict): a dictionary with printed statistics names as keys,
            and statistics statistics names to be retrieved from model.info as values.
            Default is dictionary with common statistics of stats model linear
            regression.
        append_notes (bool): a boolean variable for printing p value cutoff explanation
            and additional notes, if applicable. Default is True.
        notes_label (str): a sting to print as the title of the notes section, if
            applicable. Default is 'Notes'
        custom_notes (list): a list of strings for additional notes. Default is None.

    Returns:
        res_table (data frame, str or dictionary): depending on the rerturn type,
            data frame with formatted strings, a string for html or latex tables,
            or a dictionary with statistics and parameters dataframes, and strings
            for footers is returned. If the return type is a path, the function saves
            the resulting table at the given path.

    Notes:
        - Compiling LaTex tables requires the package siunitx.
        - Add \sisetup{input-symbols = ()} to your main tex file for proper
            compilation

    """
    assert isinstance(models, list), "Please, provide models as a list"
    models = [_process_model(model) for model in models]
    model_names = _get_model_names(models)
    default_col_names, default_col_groups = _get_default_column_names_and_groups(
        model_names
    )
    col_groups = _customize_col_groups(default_col_groups, custom_col_groups)
    col_names = _customize_col_names(default_col_names, custom_col_names)
    # Get mapping from group name to column position
    if col_groups:
        group_to_col_position = _create_group_to_col_position(col_groups)
    # Set some defaults:
    if not stat_keys:
        stat_keys = {
            "Observations": "n_obs",
            "R$^2$": "rsquared",
            "Adj. R$^2$": "rsquared_adj",
            "Residual Std. Error": "resid_std_err",
            "F Statistic": "fvalue",
            "show_dof": None,
        }
    # get common index as list.
    params = [mod.params for mod in models]
    com_ind = []
    for d_ in params:
        com_ind += [ind for ind in d_.index.to_list() if ind not in com_ind]
    # get list of columns  that need to be formatted.
    format_cols = ["value"]
    if show_inference:
        if confidence_intervals:
            format_cols += ["ci_lower", "ci_upper"]
        else:
            format_cols.append("standard_error")
    # reindex DataFrames with the common index and apply formatting.
    to_format = [mod.params.reindex(com_ind)[format_cols] for mod in models]
    raw_formatted = [_apply_number_format(df, number_format) for df in to_format]
    max_trail = int(max([_get_digits_after_decimal(df) for df in raw_formatted]))
    if add_trailing_zeros:
        formatted = [_apply_number_format(df, max_trail) for df in raw_formatted]
    else:
        formatted = raw_formatted
    # make list of DataFrames to convert to string series. Append pvalues to the
    # parameter DataFrames if show_stars is true.
    to_convert = []
    if show_stars:
        for df, mod in zip(formatted, models):
            to_convert.append(
                pd.concat([df, mod.params.reindex(com_ind)["p_value"]], axis=1)
            )
    else:
        to_convert = formatted
    # convert DataFrames to string series with inference and siginificance
    # information.
    to_concat = [
        _convert_frame_to_string_series(
            df,
            significance_levels,
            show_stars,
        )
        for df in to_convert
    ]
    body_df = pd.concat(to_concat, axis=1)
    body_df = _process_body_df(
        body_df=body_df,
        custom_param_names=custom_param_names,
        custom_index_names=custom_index_names,
        show_col_names=show_col_names,
        show_col_groups=show_col_groups,
        col_names=col_names,
        col_groups=col_groups,
    )
    to_concat = [
        _create_statistics_sr(
            mod,
            stat_keys,
            significance_levels,
            show_stars,
            number_format,
            add_trailing_zeros,
            max_trail,
        )
        for mod in models
    ]
    footer_df = pd.concat(to_concat, axis=1)
    footer_df.columns = body_df.columns
    if str(return_type).endswith("tex"):
        if siunitx_warning:
            warn(
                r"""Proper LaTeX compilation requires the package siunitx and adding
                   \sisetup{
                        group-digits             = false,
                        input-symbols            = (),
                        table-align-text-pre     = false,
                        table-align-text-post    = false
                    }
                    to your main tex file. To turn
                    this warning off set value of siunitx_warning = False"""
            )
        if len(models) > 2:
            if alignment_warning:
                warn(
                    """Set the value of padding to 3 or higher to avoid overlay
                        of columns. To turn this warning off set value of
                        alignment_warning = False"""
                )
        notes_tex = _generate_notes_latex(
            append_notes, notes_label, significance_levels, custom_notes, body_df
        )
        out = render_latex(
            body_df=body_df,
            footer_df=footer_df,
            right_decimals=max_trail,
            notes=notes_tex,
            render_options=render_options,
            show_index_names=show_index_names,
            show_col_names=show_col_names,
            show_col_groups=show_col_groups,
            group_to_col_position=group_to_col_position,
            padding=padding,
            show_footer=show_footer,
        )
    elif str(return_type).endswith("html"):
        footer = _generate_notes_html(
            append_notes, notes_label, significance_levels, custom_notes, body_df
        )
        out = render_html(
            body_df, footer_df, footer, render_options, custom_index_names, show_footer
        )
    elif return_type == "render_inputs":
        out = {
            "body_df": body_df,
            "footer_df": footer_df,
            "notes_tex": _generate_notes_latex(
                append_notes, notes_label, significance_levels, custom_notes, body_df
            ),
            "latex_right_alig": max_trail,
            "notes_html": _generate_notes_html(
                append_notes, notes_label, significance_levels, custom_notes, body_df
            ),
        }
    elif return_type == "dataframe":
        if show_footer:
            footer_df.index.names = body_df.index.names
            out = pd.concat([body_df.reset_index(), footer_df.reset_index()]).set_index(
                body_df.index.names
            )
        else:
            out = body_df
    else:
        raise TypeError("Invalid return type")
    if str(return_type).endswith((".html", ".tex")):
        with open(return_type, "w") as t:
            t.write(out)

    return out


def render_latex(
    body_df,
    footer_df,
    notes_tex,
    right_decimals,
    padding=1,
    render_options=None,
    show_col_names=True,
    show_col_groups=False,
    show_index_names=False,
    group_to_col_position=None,
    show_footer=True,
):
    """Return estimation table in LaTeX format as string.

    Args:
        body_df (pandas.DataFrame): DataFrame with formatted strings of parameter
            values, inferences (standard errors or confidence intervals, if
            applicable) and significance stars (if applicable).
        footer_df (pandas.DataFrame): DataFrame with formatted strings of summary
            statistics (such as number of observations, r-squared, etc.)
        notes_tex (str): The LaTex string with notes with additional information
            (e.g. mapping from pvalues to significance stars) to append to the footer
            of the estimation table string with LaTex code for the notes section.
        right_decimals (int): An integer passed to the `table-format` argument of
            siuntix tabular controls the number of figures that is reserved to the
            right of the decimal point. Impacts distancing between table columns and
            the distance between digits and non numerical parts (e.g. stars (*)) of
            cell strings. For detailed information and usage examples see:
            https://texdoc.org/serve/siunitx/0 (page 41).
        padding (int): Like right_decimals, is used for table alignment in siuntix
            table. Controls the number of figures reserved to the left from decimal
            points and thus the space to the left from each table column.
            For detailed information and usage examples see:
            https://texdoc.org/serve/siunitx/0 (page 41)
        render_options(dict): A dictionary with custom kwargs to pass to pd.to_latex(),
            to update the default options. An example is `{header: False}` that
            disables displaying column names.
        show_col_names (bool): If both show_col_names and show_col_groups are False,
            update `default_render_options` with `{'header': False}`. Default True.
        show_col_groups (bool): If both show_col_names and show_col_groups are False,
            update `default_render_options` with `{'header': False}`. Default False.
        show_index_names (bool): If True, updates `default_render_options` with
            `{'index_names': True}`. Default False.
        show_group_to_col_position (dict): A mapping from column groups to positions of
            columns in each group. Is used to draw centered midrules between column
            groups and column names. Default None.
        show_footer (bool): a boolean variable for displaying statistics, e.g. R2,
            Obs numbers. Default True.
    Returns:
        latex_str (str): the string for LaTex table script.

    """
    body_df = body_df.copy(deep=True)
    for i in range(body_df.columns.nlevels):
        body_df = body_df.rename(
            {c: "{" + c + "}" for c in body_df.columns.get_level_values(i)},
            axis=1,
            level=i,
        )
    body_df = body_df.applymap(_add_latex_syntax_around_scientfic_number_string)
    n_levels = body_df.index.nlevels
    n_columns = len(body_df.columns)
    # here you add all arguments of df.to_latex for which you want to change the default
    default_options = {
        "index_names": False,
        "escape": False,
        "na_rep": "",
        "column_format": "l" * n_levels
        + "S[table-format ={}.{},table-space-text-post={{-**}}]".format(
            padding, right_decimals
        )
        * n_columns,
        "multicolumn_format": "c",
    }
    if render_options:
        default_options.update(render_options)
    if show_index_names:
        default_options.update({"index_names": True})
    if not (show_col_names and show_col_groups):
        default_options.update({"header": False})
    if not default_options["index_names"]:
        body_df.index.names = [None] * body_df.index.nlevels
    latex_str = body_df.to_latex(**default_options)
    if group_to_col_position:
        temp_str = "\n"
        for k in group_to_col_position:
            max_col = max(group_to_col_position[k]) + n_levels + 1
            min_col = min(group_to_col_position[k]) + n_levels + 1
            temp_str += f"\\cmidrule(lr){{{min_col}-{max_col}}}"
            temp_str += "\n"
        latex_str = (
            latex_str.split("\\\\", 1)[0]
            + "\\\\"
            + temp_str
            + latex_str.split("\\\\", 1)[1]
        )
    latex_str = latex_str.split("\\bottomrule")[0]
    if show_footer:
        if "Observations" in footer_df.index.get_level_values(0):
            footer_df = footer_df.copy(deep=True)

            footer_df.loc["Observations"] = footer_df.loc["Observations"].applymap(
                _left_align_tex
            )
        stats_str = footer_df.to_latex(**default_options)
        stats_str = (
            "\\midrule" + stats_str.split("\\midrule")[1].split("\\bottomrule")[0]
        )
        latex_str += stats_str
    latex_str += notes_tex
    latex_str += "\\bottomrule\n\\end{tabular}\n"
    if latex_str.startswith("\\begin{table}"):
        latex_str += "\n\\end{table}\n"
    return latex_str


def render_html(
    body_df, footer_df, notes_html, render_options, custom_index_names, show_footer
):
    """Return estimation table in html format as string.

    Args:
        body_df (DataFrame): the processed dataframe with parameter values and
            precision (if applied) as strings.
        footer_df (DataFrame): the processed dataframe with summary statistics
            as strings.
        notes_html (str): a string with html code for the notes section
        render_options(dict): the pd.to_html() kwargs to apply if default options
            need to be updated.
        show_footer (bool): see main docstring

    Returns:
        html_str (str): the string for html table script.

    """
    n_levels = body_df.index.nlevels
    n_columns = len(body_df.columns)
    default_options = {"index_names": False, "na_rep": "", "justify": "center"}
    if custom_index_names:
        default_options.update({"index_names": True})
    html_str = ""
    if render_options:
        default_options.update(render_options)
        if "caption" in default_options:
            html_str += default_options["caption"] + "<br>"
            default_options.pop("caption")
    html_str += body_df.to_html(**default_options).split("</tbody>\n</table>")[0]
    # this line removes all the curly braces that were placed in order to get nice latex
    # output. Since Html does not escape them, they need to be removed.
    html_str = re.sub(
        r"(?<=[\d)}{)a-zA-Z])}", "", re.sub(r"{(?=[}\d(a-zA-Z-])", "", html_str)
    ).replace(r"\,", " ")
    if show_footer:
        stats_str = """<tr><td colspan="{}" style="border-bottom: 1px solid black">
            </td></tr>""".format(
            n_levels + n_columns
        )
        stats_str += (
            footer_df.to_html(**default_options)
            .split("</thead>\n")[1]
            .split("</tbody>\n</table>")[0]
        )
        stats_str = re.sub(r"(?<=[\d)}{)])}", "", re.sub(r"{(?=[}\d(])", "", stats_str))
        html_str += stats_str
    html_str += notes_html
    html_str += "</tbody>\n</table>"
    return html_str


def _process_model(model):
    """Check model validity, convert to namedtuple."""
    ProcessedModel = namedtuple("NamedTup", "params info name")
    if hasattr(model, "params") and hasattr(model, "info"):
        assert isinstance(model.info, dict)
        assert isinstance(model.params, pd.DataFrame)
        info = model.info
        params = model.params.copy(deep=True)
        if hasattr(model, "name"):
            assert isinstance(model.name, str)
            name = model.name
        else:
            name = ""

    else:
        if isinstance(model, dict):
            params = model["params"].copy(deep=True)
            info = model.get("info", {})
            name = model.get("name", "")
        elif isinstance(model, pd.DataFrame):
            params = model.copy(deep=True)
            info = {}
            name = ""
        else:
            try:
                params = _extract_params_from_sm(model)
                info = {**_extract_info_from_sm(model)}
                name = info.pop("name")
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise TypeError("Model {} does not have valid format".format(model))
    if "pvalue" in params.columns:
        params = params.rename(columns={"pvalue": "p_value"})
    processed_model = ProcessedModel(params=params, info=info, name=name)
    return processed_model


def _get_model_names(processed_models):
    names = []
    for i, mod in enumerate(processed_models):
        if mod.name:
            names.append(mod.name)
        else:
            names.append(f"({i + 1})")
    _check_order_of_model_names(names)
    return names


def _get_default_column_names_and_groups(model_names):
    if len(set(model_names)) == len(model_names):
        col_groups = None
        col_names = model_names
    else:
        col_groups = model_names
        col_names = [f"({i + 1})" for i in range(len(model_names))]

    return col_names, col_groups


def _customize_col_groups(custom_col_groups, default_col_groups):
    if custom_col_groups:
        if not default_col_groups:
            assert isinstance(
                custom_col_groups, list
            ), """With unique model names, multiple models can't be grouped under
            common group name. Provide list of unique group names instead, if you
            wish to add column level."""
            col_groups = custom_col_groups
        else:
            if isinstance(custom_col_groups, list):
                col_groups = custom_col_groups
            elif isinstance(custom_col_groups, dict):
                for old_key, new_key in custom_col_groups.items():
                    default_col_groups[new_key] = default_col_groups.pop(old_key)
                col_groups = default_col_groups
            else:
                raise TypeError(
                    """Invalid type for custom_col_groups. Can be either list
                    or dictionary, or NoneType"""
                )
    else:
        col_groups = default_col_groups
    return col_groups


def _customize_col_names(default_col_names, custom_col_names):
    if not custom_col_names:
        col_names = default_col_names
    elif isinstance(custom_col_names, dict):
        col_names = list(pd.Series(custom_col_names).replace(custom_col_names))
    elif isinstance(custom_col_names, list):
        col_names = custom_col_names
    else:
        raise TypeError(
            """Invalid type for custom_col_names.
            Can be either list or dictionary, or NoneType"""
        )
    return col_names


def _check_order_of_model_names(model_names):
    group_to_col_index = _create_group_to_col_position(model_names)
    for positions in group_to_col_index.values():
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                "If there are repetitions in model_names, models with the "
                f"same name need to be adjacent. You provided: {model_names}"
            )


def _create_group_to_col_position(column_groups):
    group_to_col_index = {group: [] for group in list(set(column_groups))}
    for i, group in enumerate(column_groups):
        group_to_col_index[group].append(i)
    return group_to_col_index


def _convert_frame_to_string_series(
    df,
    significance_levels,
    show_stars,
):
    """Return processed value series with significance stars and inference information.

    Args:

        df (DataFrame): params DataFrame of the model
        significance_levels (list): see main docstring
        number_format (int): see main docstring
        show_inference (bool): see main docstring
        confidence_intervals (bool): see main docstring
        show_stars (bool): see main docstring

    Returns:
        sr (pd.Series): string series with values and inferences.
    """
    value_sr = df["value"]
    if show_stars:
        sig_bins = [-1] + sorted(significance_levels) + [2]
        value_sr += "$^{"
        value_sr += (
            pd.cut(
                df["p_value"],
                bins=sig_bins,
                labels=[
                    "*" * (len(significance_levels) - i)
                    for i in range(len(significance_levels) + 1)
                ],
            )
            .astype("str")
            .replace("nan", "")
            .replace(np.nan, "")
        )
        value_sr += " }$"
    if "ci_lower" in df:
        ci_lower = df["ci_lower"]
        ci_upper = df["ci_upper"]
        inference_sr = "{("
        inference_sr += ci_lower
        inference_sr += r"\,;\,"
        inference_sr += ci_upper
        inference_sr += ")}"
        sr = _combine_series(value_sr, inference_sr)
    elif "standard_error" in df:
        standard_error = df["standard_error"]
        inference_sr = "(" + standard_error + ")"
        sr = _combine_series(value_sr, inference_sr)
    else:
        sr = value_sr
    # replace empty braces with empty string
    sr = sr.where(sr.apply(lambda x: bool(re.search(r"\d", x))), "")
    sr.name = ""
    return sr


def _combine_series(value_sr, inference_sr):
    """Merge value and inference series. Return string series
    with parameter values and precision values below respective param
    values.

    Args:
        values_sr (Series): string series of estimated parameter values
        inference_sr (Series): string series of inference values
    Returns:
        series: combined string series of param and inference values.


    """

    value_df = value_sr.to_frame(name="")
    original_cols = value_df.columns
    value_df.reset_index(drop=False, inplace=True)
    index_names = [item for item in value_df.columns if item not in original_cols]
    # set the index to even numbers, starting at 0
    value_df.index = value_df.index * 2
    inference_df = inference_sr.to_frame(name="")
    inference_df.reset_index(drop=False, inplace=True)
    # set the index to odd numbers, starting at 1
    inference_df.index = (inference_df.index * 2) + 1
    inference_df[index_names[-1]] = ""
    df = pd.concat([value_df, inference_df]).sort_index()
    df.set_index(index_names, inplace=True, drop=True)
    return df[""]


def _create_statistics_sr(
    model,
    stat_keys,
    significance_levels,
    show_stars,
    number_format,
    add_trailing_zeros,
    max_trail,
):
    """Process statistics values, return string series.

    Args:
        model (estimation result): see main docstring
        stat_keys (dict): see main docstring
        significance_levels (list): see main docstring
        show_stars (bool): see main docstring
        number_format (int): see main focstring

    Returns:
        series: string series with summary statistics values and additional info
            if applicable.

    """
    stat_values = {}
    stat_keys = deepcopy(stat_keys)
    if "show_dof" in stat_keys:
        show_dof = stat_keys.pop("show_dof")
    else:
        show_dof = None
    for k in stat_keys:
        if not stat_keys[k] == "n_obs":
            stat_values[k] = model.info.get(stat_keys[k], np.nan)
    raw_formatted = _apply_number_format(
        pd.DataFrame(pd.Series(stat_values)), number_format
    )
    if add_trailing_zeros:
        formatted = _apply_number_format(raw_formatted, max_trail)
    else:
        formatted = raw_formatted
    stat_values = formatted.to_dict()[0]
    if "n_obs" in stat_keys.values():
        n_obs = model.info.get("n_obs", np.nan)
        if not np.isnan(n_obs):
            n_obs = int(n_obs)
        stat_values[
            list(stat_keys.keys())[list(stat_keys.values()).index("n_obs")]
        ] = n_obs

    if "fvalue" in model.info and "F Statistic" in stat_values:
        if show_stars and "f_pvalue" in model.info:
            sig_bins = [-1] + sorted(significance_levels) + [2]
            sig_icon_fstat = "*" * (
                len(significance_levels)
                - np.digitize(model.info["f_pvalue"], sig_bins)
                + 1
            )
            stat_values["F Statistic"] = (
                stat_values["F Statistic"] + "$^{" + sig_icon_fstat + "}$"
            )
        if show_dof:
            fstat_str = "{{{}(df={};{})}}"
            stat_values["F Statistic"] = fstat_str.format(
                stat_values["F Statistic"],
                int(model.info["df_model"]),
                int(model.info["df_resid"]),
            )
    if "resid_std_err" in model.info and "Residual Std. Error" in stat_values:
        if show_dof:
            rse_str = "{{{}(df={})}}"
            stat_values["Residual Std. Error"] = rse_str.format(
                stat_values["Residual Std. Error"], int(model.info["df_resid"])
            )
    stat_sr = pd.Series(stat_values)
    # the follwing is to make sure statistics dataframe has as many levels of
    # indices as the parameters dataframe.
    stat_ind = np.empty((len(stat_sr), model.params.index.nlevels - 1), dtype=str)
    stat_ind = np.concatenate(
        [stat_sr.index.values.reshape(len(stat_sr), 1), stat_ind], axis=1
    ).T
    stat_sr.index = pd.MultiIndex.from_arrays(stat_ind)
    return stat_sr.astype("str").replace("nan", "")


def _process_body_df(
    df,
    custom_param_names,
    custom_index_names,
    show_col_names,
    show_col_groups,
    col_names,
    col_groups,
):
    """Process body DataFrame, customize the header.

    Args:
        df (DataFrame): string DataFrame with parameter values and inferences.
        custom_param_names (dict): see main docstring
        custom_index_names (list): see main docstring
        show_col_names (bool): see main docstring
        custom_col_names (list): see main docstring
        custom_model_names (dict): see main docstring

    Returns:
        processed_df (DataFrame): string DataFrame with customized header.

    """
    if show_col_names:
        if show_col_groups:
            df.columns = pd.MultiIndex.from_tuples(
                [(i, j) for i, j in zip(col_groups, col_names)]
            )
        else:
            df.columns = col_names
    if custom_index_names:
        if isinstance(custom_index_names, list):
            df.index.names = custom_index_names
        elif isinstance(custom_index_names, dict):
            df.rename_axis(index=custom_index_names, inplace=True)
        else:
            TypeError(
                """Invalid custom_index_names types.
                Can either be list or dict, or NoneType"""
            )
    if custom_param_names:
        ind = df.index.to_frame()
        ind = ind.replace(custom_param_names)
        df.index = pd.MultiIndex.from_frame(ind)
    return df


def _generate_notes_latex(
    append_notes, notes_label, significance_levels, custom_notes, df
):
    """Generate the LaTex script of the notes section.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        significance_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with LaTex script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    significance_levels = sorted(significance_levels)
    notes_text = "\\midrule\n"
    if append_notes:
        notes_text += "\\textit{{{}}} & \\multicolumn{{{}}}{{r}}{{".format(
            notes_label, str(n_columns + n_levels - 1)
        )
        # iterate over penultimate sig_level since last item of legend is not
        # followed by a semi column
        for i in range(len(significance_levels) - 1):
            star = "*" * (len(significance_levels) - i)
            notes_text += "$^{{{}}}$p$<${};".format(star, str(significance_levels[i]))
        notes_text += "$^{*}$p$<$" + str(significance_levels[-1]) + "} \\\\\n"
        if custom_notes:
            amp_n = "&" * n_levels
            if isinstance(custom_notes, list):
                assert all(
                    isinstance(n, str) for n in custom_notes
                ), "Data type of custom notes can only be string"
                for n in custom_notes:
                    notes_text += """
                    {}\\multicolumn{{{}}}{{r}}\\textit{{{}}}\\\\\n""".format(
                        amp_n, n_columns, n
                    )
            elif isinstance(custom_notes, str):
                notes_text += "{}\\multicolumn{{{}}}{{r}}\\textit{{{}}}\\\\\n".format(
                    amp_n, n_columns, custom_notes
                )
            else:
                raise ValueError(
                    "Custom notes can be either a string or a list of strings"
                )
    return notes_text


def _generate_notes_html(
    append_notes, notes_label, significance_levels, custom_notes, df
):
    """Generate the html script of the notes section of the estimation table.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        significance_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with html script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    significance_levels = sorted(significance_levels)
    notes_text = """<tr><td colspan="{}" style="border-bottom: 1px solid black">
        </td></tr>""".format(
        n_columns + n_levels
    )
    if append_notes:
        notes_text += """
        <tr><td style="text-align: left">{}</td><td colspan="{}"
        style="text-align: right">""".format(
            notes_label, n_columns + n_levels - 1
        )
        for i in range(len(significance_levels) - 1):
            stars = "*" * (len(significance_levels) - i)
            notes_text += "<sup>{}</sup>p&lt;{}; ".format(stars, significance_levels[i])
        notes_text += """<sup>*</sup>p&lt;{} </td>""".format(significance_levels[-1])
        if custom_notes:
            if isinstance(custom_notes, list):
                assert all(
                    isinstance(n, str) for n in custom_notes
                ), "Data type of custom notes can only be string"
                notes_text += """
                    <tr><td></td><td colspan="{}"style="text-align: right">{}</td></tr>
                    """.format(
                    n_columns + n_levels - 1, custom_notes[0]
                )
                if len(custom_notes) > 1:
                    for i in range(1, len(custom_notes)):
                        notes_text += """
                        <tr><td></td><td colspan="{}"style="text-align: right">
                        {}</td></tr>
                        """.format(
                            n_columns + n_levels - 1, custom_notes[i]
                        )
            elif isinstance(custom_notes, str):
                notes_text += """
                    <tr><td></td><td colspan="{}"style="text-align: right">{}</td></tr>
                    """.format(
                    n_columns + n_levels - 1, custom_notes
                )
            else:
                raise ValueError(
                    "Custom notes can be either a string or a list of strings"
                )

    return notes_text


def _extract_params_from_sm(model):
    """Convert statsmodels like estimation result to estimagic like params dataframe."""
    to_concat = []
    params_list = ["params", "pvalues", "bse"]
    for col in params_list:
        to_concat.append(getattr(model, col))
    to_concat.append(model.conf_int())
    params_df = pd.concat(to_concat, axis=1)
    params_df.columns = ["value", "p_value", "standard_error", "ci_lower", "ci_upper"]
    return params_df


def _extract_info_from_sm(model):
    """Process statsmodels estimation result to retrieve summary statistics as dict."""
    info = {}
    key_values = [
        "rsquared",
        "rsquared_adj",
        "fvalue",
        "f_pvalue",
        "df_model",
        "df_resid",
    ]
    for kv in key_values:
        info[kv] = getattr(model, kv)
    info["name"] = model.model.endog_names
    info["resid_std_err"] = np.sqrt(model.scale)
    info["n_obs"] = model.df_model + model.df_resid + 1
    return info


def _apply_number_format(df, number_format):
    processed_format = _process_number_format(number_format)
    if isinstance(processed_format, (list, tuple)):
        df_formatted = df.copy(deep=True).astype("float")
        for formatter in processed_format[:-1]:
            df_formatted = df_formatted.applymap(formatter.format).astype("float")
        df_formatted = df_formatted.astype("float").applymap(
            processed_format[-1].format
        )
    elif isinstance(processed_format, str):
        df_formatted = df.applymap(
            partial(_format_non_scientific_numbers, format_string=processed_format)
        )
    elif callable(processed_format):
        df_formatted = df.applymap(processed_format)
    return df_formatted


def _format_non_scientific_numbers(number_string, format_string):
    if "e" in number_string:
        out = number_string
    else:
        out = format_string.format(float(number_string))
    return out


def _process_number_format(raw_format):
    if isinstance(raw_format, str):
        processed_format = [raw_format]
    elif isinstance(raw_format, int):
        processed_format = f"{{0:.{raw_format}f}}"
    elif callable(raw_format) or isinstance(raw_format, (list, tuple)):
        processed_format = raw_format
    else:
        raise TypeError("Invalid number format")
    return processed_format


def _get_digits_after_decimal(df):
    max_trail = 0
    for c in df.columns:
        try:
            trail_length = (
                (
                    df[c][~df[c].astype("str").str.contains("e")]
                    .astype("str")
                    .str.split(".", expand=True)[1]
                    .astype("str")
                    .replace("None", "")
                )
                .str.len()
                .max()
            )
        except KeyError:
            trail_length = 0
        if trail_length > max_trail:
            max_trail = trail_length
    return max_trail


def _add_latex_syntax_around_scientfic_number_string(string):
    if "e" not in string:
        out = string
    else:
        prefix, *num_parts, suffix = re.split(r"([+-.\d+])", string)
        number = "".join(num_parts)
        out = f"{prefix}{{{number}}}{suffix}"
    return out


def _left_align_tex(n_obs):
    return f"\\multicolumn{{1}}{{l}}{{{n_obs}}}"
