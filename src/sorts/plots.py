"""
Misc. plots
"""

import logging, typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.population import Population
from sorts.space_object import SpaceObject
from sorts.plotting_deps import lp, bp, bokeh_models, pyproj
from sorts.types import (
    EcefStates,
    Float64_as_deg,
    Datetime_like,
    Datetime64_us,
    Timedelta64_us,
    Float64_as_sec,
)
from sorts.utils import to_datetime64_us
from sorts.frames import ITRS_to_geodetic
from sorts.schedule_v2 import Schedule

logger = logging.getLogger(__name__)


# TODO: re-eval if we need this
def space_object_population_table_plot(population: Population):
    source = bokeh_models.ColumnDataSource(data={f: population[f] for f in population.fields})
    columns = [bokeh_models.TableColumn(field=f, title=f) for f in population.fields]
    data_table = bokeh_models.DataTable(
        source=source,
        columns=columns,
        width=800,
        height=300,
        # selectable=True,
        # selectable="checkbox",
    )

    return data_table


# TODO: add time based binning and aggregation
# TODO: might not work that well for rx schedule, because they might have multiple pointings per slices
# TODO: also plot pointings?
def schedule_plot(
    schedule: Schedule, start_time: datetime | None = None, end_time: datetime | None = None
):
    """
    Note:
    Plotting the full schedule can be computationally demanding and lead to application crashes.
    Limiting the plot range by `start_time` and `end_time` param is recommended.

    Without aggregations, a good starting point is a 5 minutes time range.
    """

    df = schedule.to_dataframe()

    start_time = start_time if start_time is not None else df[schedule.cn.start_time].min()
    end_time = end_time if end_time is not None else df[schedule.cn.start_time].max()
    df = df[(df[schedule.cn.start_time] >= start_time) & (df[schedule.cn.end_time] <= end_time)]

    # bokeh requires str type for categorical axis
    df[schedule.cn.exp_num] = df[schedule.cn.exp_num].astype(str)

    bar = bp.figure(
        y_range=df[schedule.cn.exp_num].unique(),  # type: ignore
        x_axis_type="datetime",
        x_axis_location="above",
        width=800,
        height=300,
    )
    bar.add_tools(bokeh_models.HoverTool())

    bar.hbar(
        y=df[schedule.cn.exp_num], left=df[schedule.cn.start_time], right=df[schedule.cn.end_time]  # type: ignore
    )
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
        # x_range=bar.x_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
    )
    minimap.x_range.range_padding = 0  # type: ignore
    minimap.x_range.bounds = "auto"  # type: ignore

    # NOTE: a dummy line is plotted; select tool doesn't work well without any data plotted
    minimap.line(x=[df[schedule.cn.start_time].min(), df[schedule.cn.start_time].max()], y=[0, 0])
    minimap_range_tool = bokeh_models.RangeTool(x_range=bar.x_range, start_gesture="pan")
    minimap.add_tools(minimap_range_tool)

    plot = bp.column(bar, minimap)

    return plot


def azel_skyplot(azimuths: npt.NDArray[Float64_as_deg], elevations: npt.NDArray[Float64_as_deg]):
    """
    A plot similar to a skyplot.

    e.g. in the one in mathlab
    https://mathworks.com/help/satcom/ref/skyplot.html
    """

    # define some string const for field names
    f_azimuth = "azimuth"
    f_elevation = "elevation"
    f_adj_azimuth = "adj_azimuth"
    f_adj_elevation = "adj_elevation"

    # make a plot and set the pixel aspect ratio to equal to the data aspect ratio
    # (i.e. a circle in data will be a circle on screen)
    plot = bp.figure(match_aspect=True)
    # customize axis
    plot.xaxis.fixed_location = 0
    plot.yaxis.fixed_location = 0
    plot.xaxis.ticker = [30, 60, 90]
    plot.xaxis.major_label_overrides = {30: "60", 60: "30", 90: "0"}
    plot.yaxis.ticker = []
    plot.xaxis.axis_line_alpha = 0  # alternatively, `plot.xaxis.axis_line_color = "lightgray"`
    plot.yaxis.axis_line_alpha = 0  # alternatively, `plot.yaxis.axis_line_color = "lightgray"`

    # disable builtin grid, which is rectangular, we will draw a custom polar grid
    plot.xgrid.visible = False
    plot.ygrid.visible = False

    # add a bit more padding to the plotting region, default is 0.1 (in ratio)
    plot.x_range.range_padding = 0.15  # type: ignore
    plot.y_range.range_padding = 0.15  # type: ignore

    # disable per axis zoom when hovered on an axis
    wheelZoomTool = next(
        (t for t in plot.toolbar.tools if isinstance(t, bokeh_models.WheelZoomTool))
    )
    wheelZoomTool.zoom_on_axis = False

    # draw custom grid
    ray_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    plot.ray(
        x=[0 for _ in range(len(ray_angles))],
        y=[0 for _ in range(len(ray_angles))],
        length=90,
        angle=ray_angles,
        angle_units="deg",
        color="lightgray",
        line_width=1,
    )
    ring_sizes = [0, 30, 60, 90]  # in deg
    plot.circle(
        y=[0 for _ in range(len(ring_sizes))],
        x=[0 for _ in range(len(ring_sizes))],
        radius=ring_sizes,
        color="lightgray",
        fill_alpha=0,
        line_width=1,
    )

    # add custom labels
    for x, y, text, angle in [
        (0, 100, "N, 0°", 0),
        (100, 0, "E, 90°", -90),
        (0, -100, "S, 180°", 0),
        (-100, 0, "W, -90°", 90),
    ]:
        plot.add_layout(
            bokeh_models.Label(
                x=x,
                y=y,
                anchor="center",  # type: ignore
                text=text,
                angle=angle,
                angle_units="deg",
            )
        )

    tf = bokeh_models.PolarTransform(
        angle=f_adj_azimuth,
        radius=f_adj_elevation,
        angle_units="deg",  # type: ignore
        direction="clock",
    )

    scatter = plot.scatter(
        x=tf.x,  # type: ignore
        y=tf.y,  # type: ignore
        source={
            f_azimuth: azimuths,
            f_elevation: elevations,
            # we applied simple adjustments before plotting the data to workaround the plotting lib config limitations:
            # - start the polar 0 deg from +ve y-axis instead of + x-axis
            # - the radial range should be [90, 0] instead of [0, 90]
            f_adj_azimuth: azimuths - 90,
            f_adj_elevation: 90 - elevations,
        },
    )

    # add custom tooltips, only for the scatter plot/glyphs
    plot.add_tools(
        bokeh_models.HoverTool(
            renderers=[scatter],
            tooltips=[
                ("index", "$index"),
                ("data (az, el)", f"(@{f_azimuth}, @{f_elevation})"),
            ],
        )
    )

    return plot


# TODO: add down sampling? radar control slice are in milliseconds, while the simulation are in days or longer
def ecef_states_positions_plot(ecefs: EcefStates):
    # define some string const for field names
    f_lat = "lat"  # geodetic latitude
    f_lon = "lon"  # geodetic longitude
    f_wmx = "wmx"  # web mercator x
    f_wmy = "wmy"  # web mercator y

    geodetic_coords = ITRS_to_geodetic(ecefs[0], ecefs[1], ecefs[2])
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857"
    )  # World Geodetic System to Web Mercator
    lat = geodetic_coords[0]
    lon = geodetic_coords[1]
    wmx, wmy = transformer.transform(lat, lon)

    plot = bp.figure(
        x_axis_type="mercator",
        y_axis_type="mercator",
    )
    plot.add_tile("CartoDB Positron", retina=True)

    scatter = plot.scatter(
        x=f_wmx,  # type: ignore
        y=f_wmy,  # type: ignore
        source={
            f_lat: lat,
            f_lon: lon,
            f_wmx: wmx,
            f_wmy: wmy,
        },
    )

    # TODO: Implement a fix for the antimeridian wrapping issue
    # NOTE: We are not plotting the connecting line segments for now
    #   because bokeh does not handle correct wrapping of lines crossing antimeridian
    #   https://discourse.bokeh.org/t/bokeh-tile-antimeridian-problem/6978/3
    # plot.line(wmx, wmy)

    # add custom tooltips, only for the scatter plot/glyphs
    plot.add_tools(
        bokeh_models.HoverTool(
            renderers=[scatter],
            tooltips=[
                ("index", "$index"),
                ("data (lat, lon)", f"(@{f_lat}, @{f_lon})"),
            ],
        )
    )

    return plot


# TODO: remove?
# TODO: add down sampling? radar control slice are in milliseconds, while the simulation are in days or longer
def ecef_states_positions_plot_letsplot(ecefs: EcefStates):
    """Returns a `Dash` app, use `.run()` method to run it."""
    # plotly alternative: https://plotly.com/python/lines-on-maps/

    geodetic_coords = ITRS_to_geodetic(ecefs[0], ecefs[1], ecefs[2])
    lat = geodetic_coords[0]
    lon = geodetic_coords[1]
    latlon_df = pd.DataFrame({"lat": lat, "lon": lon})
    latlon_df = latlon_df.reset_index()
    latlon_df["lat_head"] = latlon_df["lat"].shift(-1)
    latlon_df["lon_head"] = latlon_df["lon"].shift(-1)

    # alternatively, `geom_map` can be used instead of `geom_livemap`
    #   `world_countries = lp_geo_data.geocode_countries().get_boundaries(resolution=1)`
    #   `lp.geom_map(map=world_countries, projection="epsg3857", fill='gray', color='very_light_grey', size=0.1) # mercator projection`
    plot = (
        lp.ggplot(latlon_df, lp.aes(x="lon", y="lat"))
        # map background
        + lp.geom_livemap(projection="epsg3857", zoom=1)  # mercator projection
        # plot data points as point on map
        + lp.geom_point(size=0.5, tooltips=lp.layer_tooltips().line("#: @index, lat:^y, lon:^x"))
        # also connect the points using line segments for easier reading
        + lp.geom_segment(lp.aes(xend="lon_head", yend="lat_head"), size=0.2)
        # mark the orbit direction on the 1st data point
        + lp.geom_segment(
            lp.aes(xend="lon_head", yend="lat_head"),
            data={"lat": lat[0:1], "lon": lon[0:1], "lat_head": lat[1:2], "lon_head": lon[1:2]},
            size=1,
            arrow=lp.arrow(),
        )
        # other settings
        + lp.ggtitle("World Map (In Mercator projection)")
        + lp.ggsize(800, 600)
    )

    return plot


def kepler_space_object_on_map(
    space_object: SpaceObject,
    epoch: Datetime_like,
    num_points=500,
    start_time: Datetime_like | None = None,
    end_time: Datetime_like | None = None,
):
    """Plot a space object with keplerian orbit on a map in mercator projection"""

    # TODO: check if space_object.orbit.period can be adj to always return a float
    _period = space_object.orbit.period
    period: np.timedelta64
    match _period:
        case np.ndarray():
            period = _period[0].astype("timedelta64[s]")
        case int() | float():
            period = np.timedelta64(int(_period), "s")
        case _:
            raise RuntimeError(
                f"`space_object.orbit.period` have to be of type float or ndarray of float64, but is in type {type(space_object.orbit.period)}"
            )

    period = np.timedelta64(period, "s")
    epoch = to_datetime64_us(epoch)
    start_time = to_datetime64_us(start_time) if start_time is not None else epoch
    end_time = to_datetime64_us(end_time) if end_time is not None else start_time + period

    time_arr: npt.NDArray[Datetime64_us] = np.linspace(
        start_time.astype(np.float64),
        end_time.astype(np.float64),
        num_points,
    ).astype("datetime64[us]")

    dt_arr: npt.NDArray[Timedelta64_us] = time_arr - to_datetime64_us(epoch)
    dsec_arr: npt.NDArray[Float64_as_sec] = dt_arr.astype(np.float64) / 1e6  # type: ignore

    ecefs = space_object.get_state(dsec_arr)
    plot = ecef_states_positions_plot(ecefs)

    return plot
