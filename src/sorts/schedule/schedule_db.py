from __future__ import annotations
import logging, typing as t, pathlib, sqlite3
from collections import OrderedDict
import pandas as pd
from sorts import types, utils, radar, passage, schedule
from . import schedule_dataframe
from .types import TxRxPointingPairsKey, TxRxPointingPairs


logger = logging.getLogger(__name__)


ScheduleDbConnection = t.NewType("ScheduleDbConnection", sqlite3.Connection)


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
        dfs: list[schedule_dataframe.ScheduleDataframe],
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

    def add_dataframe(self, df: schedule_dataframe.ScheduleDataframe, name: str):
        """
        Insert the dataframe as a table in DB.
        Existing table with the same name will be replaced.

        Args:
            name: Will be used as DB table name.
        """

        df.to_sql(name, self._db, if_exists="replace", index=False)
        self._db.commit()
        self.dataframe_names.update([(name, None)])

    def get_dataframe(self, name: str) -> schedule_dataframe.ScheduleDataframe:
        """Read the a table by name from DB into `ScheduleDataframe`."""

        df = pd.read_sql_query(
            f"SELECT * FROM {name}",
            self._db,
            dtype=schedule_dataframe.scheduleDataframeDtypes,
        )

        return schedule_dataframe.ScheduleDataframe(df)

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
            priorities: The list of priority corresponding to the table names.
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

    def get_tx_rx_pointing_pairs(
        self,
        start_time: types.Datetime_Like,
        end_time: types.Datetime_Like,
        tx_stn_num: int,
        rx_stn_num: int,
    ) -> TxRxPointingPairs:
        """Get pointing pairs from DB as specified by param."""

        _K = TxRxPointingPairsKey

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
            dtype={
                _K.exp_num: "int16",
                _K.rx_simult_num: "int16",
                _K.time: "datetime64[us]",
                _K.tx_pointing_e: "float64",
                _K.tx_pointing_n: "float64",
                _K.tx_pointing_u: "float64",
                _K.rx_pointing_e: "float64",
                _K.rx_pointing_n: "float64",
                _K.rx_pointing_u: "float64",
            },
        )

        return TxRxPointingPairs(df)

    def gather_pointing_pairs(
        self,
        passages: list[passage.Passage],
    ) -> dict[tuple[radar.StationId, radar.StationId], schedule.TxRxPointingPairs]:
        """
        Find the unique tx-rx station pairs among the `passages`,
        then for each pair, gather a `TxRxPointingPairs` from the schedule when the passages pass over the them.

        The returned `TxRxPointingPairs` are sorted by time in ascending order.
        """

        passages_by_tx_rx_stn_pair = passage.group_passages_by_tx_rx_station_pair(passages)

        pointing_pairs_dict: dict[
            tuple[radar.StationId, radar.StationId], schedule.TxRxPointingPairs
        ] = {}
        for stn_id_pair, passages in passages_by_tx_rx_stn_pair.items():
            pairs_ls = [
                self.get_tx_rx_pointing_pairs(
                    start_time=ps.time_range[0],
                    end_time=ps.time_range[1],
                    tx_stn_num=stn_id_pair[0],
                    rx_stn_num=stn_id_pair[1],
                )
                for ps in passages
            ]

            pairs = pd.concat(pairs_ls)
            pairs = pairs.sort_values(
                by=TxRxPointingPairsKey.time, ascending=True, ignore_index=True
            )

            pointing_pairs_dict[stn_id_pair] = TxRxPointingPairs(pairs)

        return pointing_pairs_dict
