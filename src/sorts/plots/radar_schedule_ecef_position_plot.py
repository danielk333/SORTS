import numpy.typing as npt
import pandas as pd
import bokeh.models as bokeh_models
import bokeh.layouts as bokeh_layouts
from sorts import scheduling
from sorts.types import EcefStates, Datetime64_us
from sorts.scheduling import Schedule
from .schedule_plot import _schedule_plot_from_cds
from .azel_skyplot import _azel_skyplot_cds_cols, _azel_skyplot_from_cds
from .ecef_states_positions_plot import (
    _ecef_states_positions_plot_cds_cols,
    _ecef_states_positions_plot_from_cds,
)


def _radar_schedule_ecef_position_plot_cds_df(
    ecefs: EcefStates, ecefs_time: npt.NDArray[Datetime64_us], sch: Schedule
):
    _SK = scheduling._K

    df = scheduling.to_dataframe(sch)

    # bokeh requires str type for categorical axis
    df[_SK.exp_num] = df[_SK.exp_num].astype(str)

    # insert columns for azel_skyplot
    azel_skyplot_cols = _azel_skyplot_cds_cols(
        sch[_SK.pointing][0].to_numpy(), sch[_SK.pointing][1].to_numpy()
    )
    for k, v in azel_skyplot_cols.items():
        df[k] = v

    # merging schedule and ecefs into the same dataframe
    ecef_pos_plot_cols = _ecef_states_positions_plot_cds_cols(ecefs)
    df = pd.merge(
        df,
        pd.DataFrame({_SK.start_time: ecefs_time, **ecef_pos_plot_cols}),
        on=_SK.start_time,
        how="outer",
    )

    return df


def radar_schedule_ecef_position_plot(
    ecefs: EcefStates, ecefs_time: npt.NDArray[Datetime64_us], sch: Schedule
):
    _SK = scheduling._K

    df = _radar_schedule_ecef_position_plot_cds_df(ecefs=ecefs, ecefs_time=ecefs_time, sch=sch)

    cds = bokeh_models.ColumnDataSource(df)

    start_time = df[_SK.start_time].min()
    end_time = df[_SK.end_time].max()

    sch_plot, sch_plot_bar, *_ = _schedule_plot_from_cds(
        cds,
        start_time=start_time,
        end_time=end_time,
        y_range=df[_SK.exp_num].dropna().unique(),
    )
    sch_plot_bar_select_tool = bokeh_models.BoxSelectTool()
    sch_plot_bar.add_tools(sch_plot_bar_select_tool)
    sch_plot_bar.toolbar.active_drag = sch_plot_bar_select_tool

    skyplot = _azel_skyplot_from_cds(cds)
    skyplot_select_tool = bokeh_models.LassoSelectTool()
    skyplot.add_tools(skyplot_select_tool)
    skyplot.toolbar.active_drag = skyplot_select_tool

    ecefpos_plot = _ecef_states_positions_plot_from_cds(cds)
    ecefpos_plot_select_tool = bokeh_models.LassoSelectTool()
    ecefpos_plot.add_tools(ecefpos_plot_select_tool)

    plot = bokeh_layouts.layout(
        [
            [sch_plot],
            [skyplot, ecefpos_plot],
        ]  # type: ignore
    )

    return plot
