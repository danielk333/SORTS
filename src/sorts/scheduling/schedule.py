"""
Defines the NewType `Schedule` and functions for its functionalities
"""

from __future__ import annotations
import logging, typing as t, enum, pathlib, sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas._typing as pdt
from sorts import types, utils, radar


logger = logging.getLogger(__name__)


class ScheduleKey(enum.StrEnum):
    index = "index"  # type: ignore ; seems type checker might confuse this with the `index` method from `str`
    exp_num = "exp_num"
    stn_num = "stn_num"
    simult_num = "simult_num"
    start_time = "start_time"
    end_time = "end_time"
    pointing_e = "pointing_e"
    pointing_n = "pointing_n"
    pointing_u = "pointing_u"


class _K:
    """Internal helper class for accessing string keys consistently"""

    multi_index: t.Final = "multi_index"
    start_time: t.Final = "start_time"
    exp_num: t.Final = "exp_num"
    stn_num: t.Final = "stn_num"
    simult_num: t.Final = "simult_num"
    enu: t.Final = "enu"
    e: t.Final = "e"
    n: t.Final = "n"
    u: t.Final = "u"
    end_time: t.Final = "end_time"
    pointing: t.Final = "pointing"


SimultaneousNum = int
"""An int16 that corresponds to the order in simultaneous pointings"""

ExperimentId = int
"""A unique int16 that identifies an experiment"""


@dataclass(kw_only=True)
class ExperimentDetail:
    id: ExperimentId

    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float

    slice_duration: types.Timedelta64_us
    "Duration of a control slice, in micro-second"


ExperimentDetailMap = dict[ExperimentId, ExperimentDetail]


ExperimentIdStationIdPairsMap = dict[ExperimentId, list[tuple[radar.StationId, radar.StationId]]]


ScheduleDataframe = t.NewType("ScheduleDataframe", pd.DataFrame)
"""
A pandas `DataFrame` which:
- Has an index without name or named as `"index"`
- Contains all the following columns
    ```
    - "exp_num":     np.int16
    - "stn_num":     np.int16
    - "simult_num":  np.int16
    - "start_time":  "datetime64[us]"
    - "end_time":    "datetime64[us]"
    - "pointing_e":  np.float64
    - "pointing_n":  np.float64
    - "pointing_u":  np.float64
    ```

The keys are available as enum `ScheduleKey` for consistent access.

NOTE: This is intended as an internal constructor, please use the constructor functions to create instances.
"""

ScheduleDbConnection = t.NewType("ScheduleDbConnection", sqlite3.Connection)

scheduleDataframeDtypes: t.Final[dict[t.Hashable, pdt.Dtype]] = {
    ScheduleKey.exp_num: "int16",
    ScheduleKey.stn_num: "int16",
    ScheduleKey.simult_num: "int16",
    ScheduleKey.start_time: "datetime64[us]",
    ScheduleKey.end_time: "datetime64[us]",
    ScheduleKey.pointing_e: "float64",
    ScheduleKey.pointing_n: "float64",
    ScheduleKey.pointing_u: "float64",
}
"""
The dtypes of a `ScheduleDataframe` expressed in a python dict.
Useful for certain pandas IO methods.
"""


class ScheduleValidationError(Exception):
    pass


class ScheduleDb:
    """
    Contains the schedule of radar station(s), and can it related data.

    NOTE: datetime in DB are stored as ISO 8601 string,
        with space instead of `T` as separator.
    """

    combined_schedule_name: t.Final = "_combined_schedule"

    def __init__(
        self,
        db: ScheduleDbConnection,
        dataframe_names: list[str],
    ):
        """
        NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
        """

        self._db = db
        self.dataframe_names = OrderedDict.fromkeys(dataframe_names)
        """NOTE: It is an `OrderedDict` that maps to `None` because python does not have `OrderedSet` by default."""

    @classmethod
    def empty(cls, db: str | pathlib.Path | sqlite3.Connection = ":memory:") -> t.Self:
        """Constructor for empty object."""

        if isinstance(db, sqlite3.Connection):
            db_conn = db
        else:
            db_conn = sqlite3.connect(db, autocommit=True, timeout=15.0)

        db_conn.execute("PRAGMA foreign_keys = ON")
        db_conn.execute("PRAGMA journal_mode = WAL")

        return cls(db=ScheduleDbConnection(db_conn), dataframe_names=[])

    @classmethod
    def from_schedule_dataframes(
        cls,
        dfs: list[ScheduleDataframe],
        names: list[str],
        db: str | pathlib.Path | sqlite3.Connection = ":memory:",
    ) -> t.Self:
        """Create a `ScheduleDb` from a list of `ScheduleDataframe` and their names."""

        schedule_db = cls.empty(db)
        for df, name in zip(dfs, names, strict=True):
            schedule_db.add_dataframe(df, name)

        return schedule_db

    def __getstate__(self):
        """Customize pickle read."""

        state = self.__dict__.copy()

        # replace `_db` by
        # - a path string if it is a file
        # - a dump if it is an in-memory database
        match self._db.execute("PRAGMA database_list").fetchone()[2]:
            case "":
                state["_db_dump"] = "\n".join([line for line in self._db.iterdump()])
            case str() as fpath:
                state["_db_fpath"] = fpath
            case other:
                raise RuntimeError(f"Unexcepted db path value {other}.")
        state.pop("_db", None)

        return state

    def __setstate__(self, state):
        """Customize pickle write."""

        self.__dict__.update(state)

        # restore `_db`
        db_dump = state.pop("_db_dump", None)
        db_fpath = state.pop("_db_fpath", None)

        if db_dump is not None:
            db_conn = sqlite3.connect(":memory:", autocommit=True, timeout=15.0)
            db_conn.executescript(db_dump)  # executes all SQL at once
            db_conn.commit()

            self._db = db_conn

        elif db_fpath is not None:
            db_conn = sqlite3.connect(db_fpath, autocommit=True, timeout=15.0)

        else:
            raise RuntimeError(
                "Unable to restore the backing db, both `_db_dump` and `_db_fpath` are not found in the pickle."
            )

        state.pop("_db_fpath", None)

    def add_dataframe(self, df: ScheduleDataframe, name: str):
        """
        Insert the dataframe as a table in DB.
        Existing table with the same name will be replaced.

        Args:
            name: Will be used as DB table name.
        """

        df.to_sql(name, self._db, if_exists="replace", index=False)
        self._db.commit()
        self.dataframe_names.update([(name, None)])

    def get_dataframe(self, name: str) -> ScheduleDataframe:
        """Read the a table by name from DB into `ScheduleDataframe`."""

        df = pd.read_sql_query(
            f"SELECT * FROM {name}",
            self._db,
            dtype=scheduleDataframeDtypes,
        )

        return ScheduleDataframe(df)

    def remove_dataframe(self, name: str):
        """Remove a dataframe from the DB by name."""

        self._db.execute(f"DROP TABLE IF EXISTS {name}")
        self.dataframe_names.pop(name)

    def schedule_by_priority(
        self,
        names: list[str] | None = None,
        priorities: list[int] | None = None,
        start_time: types.Datetime_Like | None = None,
        end_time: types.Datetime_Like | None = None,
        combined_schedule_name: str | None = None,
    ):
        """
        Generate a combined schedule for the specified table name and store it in DB.
        The resultant schedule is stored in the table `combined_schedule_name`,
        which defaults to `self.combined_schedule_name`.

        Args:
            names: The list of table name to combine.
                If `None`, the `dataframe_names` property will be used.
                Defaults to `None`.
            priorities: The list of priority correspondign to the table names.
                Must have the same length as the `names` param.
                If `None`, a list of `[0, ...]` will be used.
                Defaults to `None`.
            combined_schedule_name: The name of the resultant schedule. Defaults to `self.combined_schedule_name`.
        """

        if names is None:
            names = list(self.dataframe_names)

        if priorities is None:
            priorities = [0 for _ in names]

        if combined_schedule_name is None:
            combined_schedule_name = self.combined_schedule_name

        sql = f"""
            DROP TABLE IF EXISTS "{combined_schedule_name}";

            CREATE TABLE "{combined_schedule_name}" AS
            WITH sch_table AS (
                SELECT
                    row_number() OVER () AS rid -- add a int id column
                    ,*
                FROM (
                    {"\nUNION ALL\n".join([
                        f'SELECT {p} AS priority, * FROM "{n}"' for n, p in zip(names, priorities)
                    ])}
                )
                WHERE TRUE -- a dummpy condiditon to make injecting additional clause below easier
                    {f"AND start_time >= '{utils.to_pydatetime(start_time).isoformat(sep=" ")}'" if start_time is not None else ""}
                    {f"AND end_time < '{utils.to_pydatetime(end_time).isoformat(sep=" ")}'" if end_time is not None else ""}
            ),
            conflicts AS (
                SELECT cp.exp_num, cp.start_time
                FROM sch_table AS og
                JOIN sch_table AS cp
                    -- on same radar station
                    ON og.stn_num = cp.stn_num
                    -- and different exp_num. (also prevent self-comparison of the same row)
                    AND og.exp_num != cp.exp_num
                WHERE og.start_time < cp.end_time
                    AND og.end_time > cp.start_time
                    AND (
                        -- lower priority number means more important
                        -- for equal priority, the first one
                        og.priority < cp.priority
                        OR (og.priority = cp.priority AND og.rid < cp.rid)
                    )
            )
            -- in the current implementation, we assume entries from all other radar stations
            -- of the same experiment have to be removed as well
            SELECT
                exp_num, stn_num, simult_num
                ,start_time, end_time
                ,pointing_e, pointing_n, pointing_u
            FROM sch_table
            WHERE (exp_num, start_time) NOT IN (SELECT exp_num, start_time FROM conflicts)
            ORDER BY start_time ASC, end_time ASC, simult_num ASC, stn_num ASC, exp_num ASC
            ;"""

        self._db.executescript(sql)
        self._db.commit()

    TxRxPointingPairs = t.NewType("TxRxPointingPairs", pd.DataFrame)
    """
    A pandas `Dataframe` with
    ```
    Columns:
        exp_num        int16
        rx_stn_num     int16
        rx_simult_num  int16
        time           datetime64[us]
        tx_pointing_e  float64
        tx_pointing_n  float64
        tx_pointing_u  float64
        rx_pointing_e  float64
        rx_pointing_n  float64
        rx_pointing_u  float64
    ```
    """

    def get_tx_rx_pointing_pairs(
        self,
        start_time: types.Datetime_Like,
        end_time: types.Datetime_Like,
        tx_stn_num: int,
        rx_stn_num: int,
    ) -> TxRxPointingPairs:

        df = pd.read_sql_query(
            f"""
            WITH rx_sch AS (
                SELECT *
                FROM "{self.combined_schedule_name}"
                WHERE stn_num = {rx_stn_num}
                    AND start_time >= '{utils.to_pydatetime(start_time).isoformat(sep=" ")}'
                    AND end_time < '{utils.to_pydatetime(end_time).isoformat(sep=" ")}'
            ),
            tx_sch AS (
                SELECT *
                FROM "{self.combined_schedule_name}"
                WHERE stn_num = {tx_stn_num}
                    AND start_time >= '{utils.to_pydatetime(start_time).isoformat(sep=" ")}'
                    AND end_time < '{utils.to_pydatetime(end_time).isoformat(sep=" ")}'
            )
            SELECT
                rx_sch.exp_num AS exp_num
                ,rx_sch.stn_num AS rx_stn_num
                ,rx_sch.simult_num AS rx_simult_num
                ,rx_sch.start_time AS time
                ,tx_sch.pointing_e AS tx_pointing_e
                ,tx_sch.pointing_n AS tx_pointing_n
                ,tx_sch.pointing_u AS tx_pointing_u
                ,rx_sch.pointing_e AS rx_pointing_e
                ,rx_sch.pointing_n AS rx_pointing_n
                ,rx_sch.pointing_u AS rx_pointing_u
            FROM rx_sch
                JOIN tx_sch
                ON rx_sch.start_time = tx_sch.start_time
                AND rx_sch.end_time = tx_sch.end_time
                AND rx_sch.exp_num = tx_sch.exp_num
            ;""",
            self._db,
            # TODO: replace hard-coded string key here by StrEnum
            dtype={
                "exp_num": "int16",
                "rx_stn_num": "int16",
                "rx_simult_num": "int16",
                "time": "datetime64[us]",
                "tx_pointing_e": "float64",
                "tx_pointing_n": "float64",
                "tx_pointing_u": "float64",
                "rx_pointing_e": "float64",
                "rx_pointing_n": "float64",
                "rx_pointing_u": "float64",
            },
        )

        return self.TxRxPointingPairs(df)


def validate_schedule_dataframe(df: pd.DataFrame) -> ScheduleDataframe:
    """
    Validate a pandas `DataFrame` against the definition of `ScheduleDataframe`.

    See the docs of `ScheduleDataframe` for its definition.

    Returns:
        The original DataFrame casted into a `ScheduleDataframe`.

    Raises:
        `ScheduleValidationError`
    """

    if not (df.index.name is None or df.index.name == ScheduleKey.index):
        raise ScheduleValidationError("DataFrame index name is not `None` or `'index'`")

    if not {key.value for key in ScheduleKey if key != ScheduleKey.index}.issubset(df.columns):
        raise ScheduleValidationError("One or more column is missing from the DataFrame")

    if df.dtypes[ScheduleKey.exp_num] != "int16":
        raise ScheduleValidationError("Column 'exp_num' is not numpy dtype 'int16'")
    if df.dtypes[ScheduleKey.stn_num] != "int16":
        raise ScheduleValidationError("Column 'stn_num' is not numpy dtype 'int16'")
    if df.dtypes[ScheduleKey.simult_num] != "int16":
        raise ScheduleValidationError("Column 'simult_num' is not numpy dtype 'int16'")

    if df.dtypes[ScheduleKey.start_time] != "datetime64[us]":
        raise ScheduleValidationError("Column 'start_time' is not numpy dtype 'datetime64[us]'")
    if df.dtypes[ScheduleKey.end_time] != "datetime64[us]":
        raise ScheduleValidationError("Column 'end_time' is not numpy dtype 'datetime64[us]'")

    if df.dtypes[ScheduleKey.pointing_e] != "float64":
        raise ScheduleValidationError("Column 'pointing_e' is not numpy dtype 'float64'")
    if df.dtypes[ScheduleKey.pointing_n] != "float64":
        raise ScheduleValidationError("Column 'pointing_n' is not numpy dtype 'float64'")
    if df.dtypes[ScheduleKey.pointing_u] != "float64":
        raise ScheduleValidationError("Column 'pointing_u' is not numpy dtype 'float64'")

    return ScheduleDataframe(df)


def schedule_dataframe_from_rows(rows: list[list[t.Any]]) -> ScheduleDataframe:
    """Create a `ScheduleDataframe` from rows of data."""

    # NOTE: Constructing `DataFrame` from `Series` seems to be the only safe way to ensure
    #       the datetime resolution is not overrided into `'ns'` from pandas's type infer attempt.

    if len(rows) == 0:
        cols = [[] for key in ScheduleKey if key != ScheduleKey.index]
    else:
        cols = list(zip(*rows))

    df = pd.DataFrame(
        {
            ScheduleKey.exp_num: pd.Series(cols[0], dtype=np.int16),
            ScheduleKey.stn_num: pd.Series(cols[1], dtype=np.int16),
            ScheduleKey.simult_num: pd.Series(cols[2], dtype=np.int16),
            ScheduleKey.start_time: pd.Series(cols[3], dtype="datetime64[us]"),
            ScheduleKey.end_time: pd.Series(cols[4], dtype="datetime64[us]"),
            ScheduleKey.pointing_e: pd.Series(cols[5], dtype=np.float64),
            ScheduleKey.pointing_n: pd.Series(cols[6], dtype=np.float64),
            ScheduleKey.pointing_u: pd.Series(cols[7], dtype=np.float64),
        }
    )

    return validate_schedule_dataframe(df)


def schedule_dataframe_from_series(
    exp_num: pd.Series,
    stn_num: pd.Series,
    simult_num: pd.Series,
    start_time: pd.Series,
    end_time: pd.Series,
    pointing_e: pd.Series,
    pointing_n: pd.Series,
    pointing_u: pd.Series,
) -> ScheduleDataframe:
    """Create an `ScheduleDataframe` from columns of pandas `Series`."""

    df = pd.DataFrame(
        {
            ScheduleKey.exp_num: exp_num,
            ScheduleKey.stn_num: stn_num,
            ScheduleKey.simult_num: simult_num,
            ScheduleKey.start_time: start_time,
            ScheduleKey.end_time: end_time,
            ScheduleKey.pointing_e: pointing_e,
            ScheduleKey.pointing_n: pointing_n,
            ScheduleKey.pointing_u: pointing_u,
        }
    )

    return validate_schedule_dataframe(df)


def schedule_dataframe_from_ndarrays(
    exp_num: npt.NDArray,
    stn_num: npt.NDArray,
    simult_num: npt.NDArray,
    start_time: npt.NDArray,
    end_time: npt.NDArray,
    pointing_e: npt.NDArray,
    pointing_n: npt.NDArray,
    pointing_u: npt.NDArray,
) -> ScheduleDataframe:
    """Create an empty `ScheduleDataframe` from columns of numpy `ndarray`."""

    df = pd.DataFrame(
        {
            ScheduleKey.exp_num: pd.Series(exp_num, dtype=np.int16),
            ScheduleKey.stn_num: pd.Series(stn_num, dtype=np.int16),
            ScheduleKey.simult_num: pd.Series(simult_num, dtype=np.int16),
            ScheduleKey.start_time: pd.Series(start_time, dtype="datetime64[us]"),
            ScheduleKey.end_time: pd.Series(end_time, dtype="datetime64[us]"),
            ScheduleKey.pointing_e: pd.Series(pointing_e, dtype=np.float64),
            ScheduleKey.pointing_n: pd.Series(pointing_n, dtype=np.float64),
            ScheduleKey.pointing_u: pd.Series(pointing_u, dtype=np.float64),
        }
    )

    return validate_schedule_dataframe(df)


def empty_schedule_dataframe() -> ScheduleDataframe:
    """Create an empty `ScheduleDataframe`"""

    return schedule_dataframe_from_rows([])
