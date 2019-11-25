"""Functions to create new databases or load existing ones.

Note: Most functions in this module change their arguments in place since this is the
recommended way of doing things in sqlalchemy and makes sense for database code.

"""
from pathlib import Path

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


def load_database(path):
    """Return database metadata object for the database stored in ``path``.

    This is the default way of loading a database for read-only purposes in estimagic.

    Args:
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.

    Returns:
        database (sqlalchemy.MetaData). The engine that connects to the database can be
        accessed via ``database.bind``.

    """
    if isinstance(path, str):
        path = Path(path)

    engine = create_engine(f"sqlite:///{path}")
    _make_engine_thread_safe(engine)
    database = MetaData()
    database.bind = engine
    database.reflect()
    return database


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


def prepare_database(
    path, params, db_options=None, optimization_status="scheduled", gradient_status=0,
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
    - db_options: table with one row and one column called "value". It contains
      a dictionary with the dashboard options. Internally this is a PickleType, so the
      dictionary must be pickle serializable. Initialized to db_options.


    Args:
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.
        params (pd.DataFrame): see :ref:`params`.
        db_options (dict): Dashboard options.
        optimization_status (str): One of "scheduled", "running", "success", "failure".
        gradient_status (float): Progress of gradient calculation between 0 and 1.

    Returns:
        database (sqlalchemy.sql.schema.MetaData). The engine that connects
        to the database can be accessed via ``database.bind``.

    """
    db_options = {} if db_options is None else db_options
    gradient_status = float(gradient_status)
    database = load_database(path)

    opt_tables = [
        "params_history",
        "gradient_history",
        "criterion_history",
        "timestamps",
        "convergence_history",
        "start_params",
        "optimization_status",
        "gradient_status",
        "db_options",
    ]

    for table in opt_tables:
        if table in database.tables:
            database.tables[table].drop(database.bind)

    _define_params_history_table(database, params)
    _define_gradient_history_table(database, params)
    _define_criterion_history_table(database)
    _define_time_stamps_table(database)
    _define_convergence_history_table(database)
    _define_start_params_table(database)
    _define_optimization_status_table(database)
    _define_gradient_status_table(database)
    _define_db_options_table(database)
    engine = database.bind
    database.create_all(engine)

    append_rows(database, "start_params", {"value": params})
    append_rows(database, "optimization_status", {"value": optimization_status})
    append_rows(database, "gradient_status", {"value": gradient_status})
    append_rows(database, "db_options", {"value": db_options})

    return database


def _define_params_history_table(database, params):
    names = params["name"].tolist()
    cols = [Column(name, Float) for name in names]
    parvals = Table(
        "params_history",
        database,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
    )
    return parvals


def _define_gradient_history_table(database, params):
    names = params["name"].tolist()
    cols = [Column(name, Float) for name in names]
    gradvals = Table(
        "gradient_history",
        database,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
    )
    return gradvals


def _define_criterion_history_table(database):
    critvals = Table(
        "criterion_history",
        database,
        Column("iteration", Integer, primary_key=True),
        Column("value", Float),
        sqlite_autoincrement=True,
    )
    return critvals


def _define_time_stamps_table(database):
    tstamps = Table(
        "timestamps",
        database,
        Column("iteration", Integer, primary_key=True),
        Column("value", DATETIME),
        sqlite_autoincrement=True,
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
    )
    return term


def _define_start_params_table(database):
    params_table = Table("start_params", database, Column("value", PickleType))
    return params_table


def _define_optimization_status_table(database):
    optstat = Table("optimization_status", database, Column("value", String),)
    return optstat


def _define_gradient_status_table(databes):
    gradstat = Table("gradient_status", databes, Column("value", Float),)
    return gradstat


def _define_db_options_table(database):
    db_options = Table("db_options", database, Column("value", PickleType),)
    return db_options
