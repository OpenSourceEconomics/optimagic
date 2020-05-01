"""Functions to create new databases or load existing ones.

Note: Most functions in this module change their arguments in place since this is the
recommended way of doing things in sqlalchemy and makes sense for database code.

"""
import io
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import BLOB
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import PickleType
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.dialects.sqlite import DATETIME

from estimagic.logging.update_database import append_rows


def prepare_database(metadata=None, path=None):
    """Return a bound sqlalchemy.MetaData object for the database stored in ``path``.

    This is the only acceptable way of creating or loading databases in estimagic!

    If metadata is a bound MetaData object, it is just returned. If metadata is given
    but not bound, we bind it to an engine that connects to the database stored under
    ``path``. If only the path is provided, we generate an appropriate MetaData object
    and bind it to the database.

    Args:
        metadata (sqlalchemy.MetaData): MetaData object that might or might not be
            bound to the database under path. In any case it needs to be compatible
            with the database stored under ``path``. For speed reasons, this is not
            checked.
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.

    Returns:
        metadata (sqlalchemy.MetaData). MetaData object that is bound to the database
        under ``path``.

    """
    if isinstance(path, str):
        path = Path(path)

    if isinstance(metadata, MetaData):
        if metadata.bind is None:
            assert (
                path is not None
            ), "If metadata is not bound, you need to provide a path."
            engine = create_engine(f"sqlite:///{path}")
            _make_engine_thread_safe(engine)
            metadata.bind = engine
    elif metadata is None:
        assert path is not None, "If metadata is None you need to provide a path."
        engine = create_engine(f"sqlite:///{path}")
        _make_engine_thread_safe(engine)
        metadata = MetaData()
        metadata.bind = engine
        metadata.reflect()
    else:
        raise ValueError("metadata must be sqlalchemy.MetaData or None.")

    return metadata


def _make_engine_thread_safe(engine):
    """Make the engine even more thread safe than by default.

    The code is taken from the documentation: https://tinyurl.com/u9xea5z

    The main purpose is to emit the begin statement of any connection
    as late as possible in order to keep the time in which the database
    is locked as short as possible.

    """

    @event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before absolutely necessary.
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN
        conn.execute("BEGIN DEFERRED")


def add_optimization_tables_to_database(
    path,
    params,
    comparison_plot_data=None,
    dash_options=None,
    constraints=None,
    optimization_status="scheduled",
    gradient_status=0,
):
    """Return database metadata object with all relevant tables for the optimization.

    This should always be used to create entirely new databases or to create the
    tables needed during optimization in an existing database.

    A new database is created if path does not exist yet. Otherwise the
    existing database is loaded and all tables needed to log the optimization are
    overwritten. Other tables remain unchanged.

    The resulting database has the following tables:

    - params_history: the complete history of parameters from the optimization. The
      index column is "iteration", the remaining columns are parameter names taken
      from params["name"].
    - gradient_history: the complete history of gradient evaluations from the
      optimization. Same columns as params_history.
    - criterion_history: the complete history of criterion values from the optimization.
      The index column is "iteration", the second column is "value".
    - time_stamps: timestamps from the end of each criterion evaluation. Same columns as
      criterion_history.
    - convergence_history: the complete history of convergence criteria from the
      optimization. The index column is "iteration", the other columns are "ftol",
      "gtol" and "xtol".
    - start_params: copy of user provided ``params``. This is not just the first entry
      of params_history because it contains all columns and has a different index.
    - optimization_status: table with one row and one column called "value" which takes
      the values "scheduled", "running", "success" or "failure". Initialized to
      ``optimization_status``.
    - gradient_status: table with one row and one column called "value" which
      can be any float between 0 and 1 and indicates the progress of the gradient
      calculation. Initialized to ``gradient_status``
    - dash_options: table with one row and one column called "value". It contains
      a dictionary with the dashboard options.
      Internally this is a PickleType, so the dictionary must be pickle serializable.
      Initialized to dash_options.
    - exceptions: table with one column called "value" with exceptions.
    - constraints: table with one row and one column called "value". It contains the
      list of constraints. Internally this is a PickleType, so the list must be pickle
      serializable.


    Args:
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.
        params (pd.DataFrame): see :ref:`params`.
        comparison_plot_data : (numpy.ndarray or pandas.Series or pandas.DataFrame):
            Contains the data for the comparison plot. Later updates will only deliver
            the value column where as this input has an index and other invariant
            information.
        dash_options (dict): Dictionary with the dashboard options.
        optimization_status (str): One of "scheduled", "running", "success", "failure".
        gradient_status (float): Progress of gradient calculation between 0 and 1.
        constraints (list): List of constraints.

    Returns:
        database (sqlalchemy.MetaData). The engine that connects
        to the database can be accessed via ``database.bind``.

    """
    gradient_status = float(gradient_status)
    database = prepare_database(path=path)

    opt_tables = [
        "params_history",
        "gradient_history",
        "criterion_history",
        "timestamps",
        "convergence_history",
        "start_params",
        "comparison_plot",
        "optimization_status",
        "gradient_status",
        "dash_options",
        "exceptions",
        "constraints",
    ]

    for table in opt_tables:
        if table in database.tables:
            database.tables[table].drop(database.bind)

    _define_table_formatted_with_params(database, params, "params_history")
    _define_table_formatted_with_params(database, params, "gradient_history")
    _define_fitness_history_table(database, "criterion_history")
    _define_time_stamps_table(database)
    _define_convergence_history_table(database)
    _define_start_params_table(database)
    _define_one_column_pickle_table(database, "comparison_plot")
    _define_optimization_status_table(database)
    _define_gradient_status_table(database)
    _define_scalar_pickle_table(database, "dash_options")
    _define_string_table(database, "exceptions")
    _define_scalar_pickle_table(database, "constraints")
    engine = database.bind
    database.create_all(engine)

    append_rows(database, "start_params", {"value": params})
    append_rows(database, "optimization_status", {"value": optimization_status})
    append_rows(database, "gradient_status", {"value": gradient_status})
    append_rows(database, "dash_options", {"value": dash_options})
    append_rows(database, "constraints", {"value": constraints})

    if comparison_plot_data is None:
        comparison_plot_data = pd.DataFrame({"value": [np.nan]})
    append_rows(database, "comparison_plot", {"value": comparison_plot_data})

    return database


def _define_table_formatted_with_params(database, params, table_name):
    cols = [Column(name, Float) for name in params["name"]]
    values = Table(
        table_name,
        database,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
        extend_existing=True,
    )
    return values


def _define_fitness_history_table(database, table_name):
    critvals = Table(
        table_name,
        database,
        Column("iteration", Integer, primary_key=True),
        Column("value", Float),
        sqlite_autoincrement=True,
        extend_existing=True,
    )
    return critvals


def _define_time_stamps_table(database):
    tstamps = Table(
        "timestamps",
        database,
        Column("iteration", Integer, primary_key=True),
        Column("value", DATETIME),
        sqlite_autoincrement=True,
        extend_existing=True,
    )
    return tstamps


def _define_convergence_history_table(database):
    names = ["ftol", "gtol", "xtol"]
    cols = [Column(name, Float) for name in names]
    term = Table(
        "convergence_history",
        database,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
        extend_existing=True,
    )
    return term


def _define_start_params_table(database):
    start_params_table = Table(
        "start_params", database, Column("value", PickleType), extend_existing=True
    )
    return start_params_table


def _define_one_column_pickle_table(database, table):
    params_table = Table(
        table,
        database,
        Column("iteration", Integer, primary_key=True),
        Column("value", PickleType),
        sqlite_autoincrement=True,
        extend_existing=True,
    )
    return params_table


def _define_optimization_status_table(database):
    optstat = Table(
        "optimization_status", database, Column("value", String), extend_existing=True
    )
    return optstat


def _define_gradient_status_table(database):
    gradstat = Table(
        "gradient_status", database, Column("value", Float), extend_existing=True
    )
    return gradstat


def _define_scalar_pickle_table(database, name):
    dash_options = Table(
        name, database, Column("value", PickleType), extend_existing=True
    )
    return dash_options


def _define_string_table(database, name):
    exception_table = Table(
        name, database, Column("value", String), extend_existing=True
    )

    return exception_table


def read_last_iterations(database, tables, n, return_type, path=None):
    """Read the last n iterations from all tables.

    If a table has less than n obervations, all observations are returned.

    Args:
        database (sqlalchemy.MetaData)
        tables (list): List of tables names.
        n (int): number of rows to retrieve
        return_type (str): one of "list", "pandas", "bokeh"
            - "list": A list of lists. The first sublist are the columns. The remaining
              sublists are retrieved rows.
            - "pandas": A dataframe.
            - "bokeh": A dictionary that can be used to stream to a ColumnDataSource.
              It has one key per column and the corresponding values are lists that
              contain the data of that column.
        path (str or pathlib.Path): Path to the database. Only necessary if database
            can be un-bound, e.g. if the bind argument was lost due to a pickling step
            in a parallelized optimization.

    Returns:
        result (dict or return_type):
            If ``tables`` has only one entry, return the last iterations of that table,
            converted to return_type. If ``tables`` has several entries, return a
            dictionary with one entry per table.

    """
    database = prepare_database(metadata=database, path=path)
    if isinstance(tables, (str, int)):
        tables = [tables]
    # sqlalchemy fails silently with many numpy integer types, e.g. np.int64.
    n = int(n)

    selects = []
    for table in tables:
        tab = database.tables[table]
        sel = tab.select().order_by(tab.c.iteration.desc()).limit(n)
        selects.append(sel)

    raw_results = _execute_select_statements(selects, database)
    ordered_results = [res[::-1] for res in raw_results]

    result = _process_selection_result(database, tables, ordered_results, return_type)
    return result


def read_new_iterations(
    database, tables, last_retrieved, return_type, limit=None, path=None
):
    """Read all iterations after last_retrieved.

    Args:
        database (sqlalchemy.MetaData)
        tables (list): List of tables names.
        last_retrieved (int): The last iteration that was retrieved.
        return_type (str): one of "list", "pandas", "bokeh"
        limit (int): Only the first ``limit`` rows will be retrieved. Default None.
        path (str or pathlib.Path): Path to the database. Only necessary if database
            can be un-bound, e.g. if the bind argument was lost due to a pickling step
            in a parallelized optimization.


    Returns:
        result (dict or return_type):
            If ``tables`` has only one entry, return the last iterations of that table,
            converted to return_type. If ``tables`` has several entries, return a
            dictionary with one entry per table.
        int: The new last_retrieved value.

    """
    database = prepare_database(metadata=database, path=path)
    if isinstance(tables, (str, int)):
        tables = [tables]
    # sqlalchemy fails silently with many numpy integer types, e.g. np.int64.
    last_retrieved = int(last_retrieved)
    limit = int(limit)

    selects = []
    for table in tables:
        tab = database.tables[table]
        sel = tab.select().where(tab.c.iteration > last_retrieved).limit(limit)
        selects.append(sel)

    raw_results = _execute_select_statements(selects, database)
    if len(raw_results[0]) > 0:
        new_last = raw_results[0][-1][0]
    else:
        new_last = last_retrieved
    result = _process_selection_result(database, tables, raw_results, return_type)
    return result, new_last


def read_scalar_field(database, table, path=None):
    """Read the value of a table with one row and one column called "value".

    Args:
        database (sqlalchemy.MetaData)
        table (str): Name of the table.
        path (str or pathlib.Path): Path to the database. Only necessary if database
            can be un-bound, e.g. if the bind argument was lost due to a pickling step
            in a parallelized optimization.

    """
    database = prepare_database(metadata=database, path=path)
    sel = database.tables[table].select()
    res = _execute_select_statements(sel, database)[0][0][0]
    if isinstance(database.tables[table].c.value.type, BLOB):
        res = pd.read_pickle(io.BytesIO(res), compression=None)
    return res


def _execute_select_statements(statements, database):
    """Execute a list of select statements in one atomic transaction.

    If any statement fails, the transaction is rolled back, and a warning is issued.

    Args:
        statements (list or sqlalchemy statement): List of sqlalchemy select statements.
        database (sqlalchemy.MetaData): The bind argument must be set.


    Returns:
        result (list): List of selection results. A selection result is a list of
        tuples where each tuple is a selected row.

    """
    if not isinstance(statements, (list, tuple)):
        statements = [statements]

    results = []
    engine = database.bind
    conn = engine.connect()
    # acquire lock
    trans = conn.begin()
    try:
        for stat in statements:
            res = conn.execute(stat)
            results.append(list(res))
            res.close()
        # release lock
        trans.commit()
        conn.close()
    except (KeyboardInterrupt, SystemExit):
        trans.rollback()
        conn.close()
        raise
    except Exception:
        exception_info = traceback.format_exc()
        warnings.warn(
            "Unable to read from database. Try again later. The traceback was:\n\n"
            f"{exception_info}"
        )

        trans.rollback()
        conn.close()
        results = [[] for stat in statements]

    return results


def _transpose_nested_list(nested_list):
    """Transpose a list of lists."""
    return list(map(list, zip(*nested_list)))


def _process_selection_result(database, tables, raw_results, return_type):
    """Convert sqlalchemy selection results to desired return_type."""
    result = {}
    for table, raw_res in zip(tables, raw_results):
        columns = database.tables[table].columns.keys()
        if return_type == "list":
            res = [columns]
            for row in raw_res:
                res.append(list(row))
        elif return_type == "bokeh":
            res = dict(zip(columns, _transpose_nested_list(raw_res)))
            if res == {}:
                res = {col: [] for col in columns}
        elif return_type == "pandas":
            res = pd.DataFrame(data=raw_res, columns=columns).set_index("iteration")
        result[table] = res

    if len(tables) == 1:
        result = list(result.values())[0]
    return result
