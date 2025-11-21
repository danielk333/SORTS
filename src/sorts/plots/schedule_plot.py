import typing as t
import numpy as np
import numpy.typing as npt
import bokeh.plotting as bp
import bokeh.models as bokeh_models
from sorts import scheduling
from sorts.types import Datetime_Like, Datetime64_us
from sorts.utils import to_datetime64_us
from sorts.scheduling import Schedule


def _schedule_plot_from_cds(
    source: bokeh_models.ColumnarDataSource,
    start_time: Datetime64_us,
    end_time: Datetime64_us,
    y_range: t.Sequence[str] | npt.NDArray[np.str_],
):
    """An internal ver of `schedule_plot` that takes a bokeh `ColumnDataSource`"""

    _SK = scheduling._K

    bar = bp.figure(
        y_range=y_range,  # type: ignore
        x_axis_type="datetime",
        x_axis_location="above",
        width=800,
        height=300,
    )
    bar.add_tools(bokeh_models.HoverTool())

    bar.hbar(
        y=_SK.exp_num,
        left=_SK.start_time,
        right=_SK.end_time,
        source=source,
    )
    bar.x_range.range_padding = 0  # type: ignore
    bar_xpan_tool = bokeh_models.PanTool(dimensions="width")
    bar_xwheel_zoom_tool = bokeh_models.WheelZoomTool(dimensions="width")
    bar.add_tools(bar_xpan_tool)
    bar.add_tools(bar_xwheel_zoom_tool)
    bar.toolbar.active_drag = bar_xpan_tool
    bar.toolbar.active_scroll = bar_xwheel_zoom_tool

    minimap = bp.figure(
        title="Drag the middle and edges of the selection box to change the range above",
        height=130,
        width=800,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
    )
    minimap.x_range.range_padding = 0.05  # type: ignore
    minimap.x_range.bounds = "auto"  # type: ignore

    # NOTE: a dummy line is plotted; select tool doesn't work well without any data plotted
    minimap.line(
        x=[start_time, end_time],  # type: ignore
        y=[0, 0],
    )
    minimap_range_tool = bokeh_models.RangeTool(x_range=bar.x_range, start_gesture="pan")
    minimap.add_tools(minimap_range_tool)

    plot = bp.column(bar, minimap)

    return plot, bar, minimap


# TODO: add time based binning and aggregation
def schedule_plot(
    sch: Schedule,
    start_time: Datetime_Like | None = None,
    end_time: Datetime_Like | None = None,
):
    """
    Note:
    Plotting the full schedule can be computationally demanding and lead to application crashes.
    Limiting the plot range by `start_time` and `end_time` param is recommended.

    Without aggregations, a good starting point is a 5 minutes time range.
    """

    _SK = scheduling._K

    df = scheduling.to_dataframe(sch)

    start_time_: Datetime64_us = (
        to_datetime64_us(start_time) if start_time is not None else df[_SK.start_time].min()
    )
    end_time_: Datetime64_us = (
        to_datetime64_us(end_time) if end_time is not None else df[_SK.end_time].max()
    )

    df = df[(df[_SK.start_time] >= start_time_) & (df[_SK.end_time] <= end_time_)]
    # bokeh requires str type for categorical axis
    df[_SK.exp_num] = df[_SK.exp_num].astype(str)

    plot, *_ = _schedule_plot_from_cds(
        source=bokeh_models.ColumnDataSource(df),
        start_time=start_time_,
        end_time=end_time_,
        y_range=df[_SK.exp_num].unique(),
    )

    return plot
