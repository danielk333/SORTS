"""
Shared types in this subpackage.

(Types might live in their own module instead of here if it improves readability,
and the imports can be worked around, e.g, by `if t.TYPE_CHECKING`)
"""

import typing as t
from sorts.utils import assert_class_attributes_equal_to

DataKey = t.Literal["pointing", "exp_num", "simult_num"]
CoordKey = t.Literal["start_time", "end_time", "enu", "e", "n", "u"]
AttrKey = t.Literal["stn_id", "exp_detail_map"]
Key = t.Literal[DataKey, CoordKey, AttrKey]


class _K:
    """Internal helper class for accessing string keys consistently"""

    pointing: t.Final = "pointing"
    exp_num: t.Final = "exp_num"
    simult_num: t.Final = "simult_num"
    start_time: t.Final = "start_time"
    end_time: t.Final = "end_time"
    enu: t.Final = "enu"
    e: t.Final = "e"
    n: t.Final = "n"
    u: t.Final = "u"
    stn_id: t.Final = "stn_id"
    exp_detail_map: t.Final = "exp_detail_map"


assert_class_attributes_equal_to(_K, t.get_args(Key))
