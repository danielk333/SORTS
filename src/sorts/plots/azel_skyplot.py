import typing as t
import numpy.typing as npt
import bokeh.plotting as bp
import bokeh.models as bokeh_models
from sorts.types import Float64_as_deg
from sorts.utils import assert_class_attributes_equal_to


AzelSkyplotColumnKey = t.Literal["azimuth", "elevation", "adj_azimuth", "adj_elevation"]


class _K_AzelSkyplotColumnKey:
    """Internal helper class for accessing string keys consistently"""

    azimuth: t.Final = "azimuth"
    elevation: t.Final = "elevation"
    adj_azimuth: t.Final = "adj_azimuth"
    adj_elevation: t.Final = "adj_elevation"


assert_class_attributes_equal_to(_K_AzelSkyplotColumnKey, t.get_args(AzelSkyplotColumnKey))


def _azel_skyplot_from_cds(source: bokeh_models.ColumnarDataSource):
    """An internal ver of `azel_skyplot` that takes a bokeh `ColumnDataSource`."""

    _K = _K_AzelSkyplotColumnKey

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
        angle=_K.adj_azimuth,
        radius=_K.adj_elevation,
        angle_units="deg",  # type: ignore
        direction="clock",
    )

    scatter = plot.scatter(
        x=tf.x,  # type: ignore
        y=tf.y,  # type: ignore
        source=source,
    )

    # add custom tooltips, only for the scatter plot/glyphs
    plot.add_tools(
        bokeh_models.HoverTool(
            renderers=[scatter],
            tooltips=[
                ("index", "$index"),
                ("data (az, el)", f"(@{_K.azimuth}, @{_K.elevation})"),
            ],
        )
    )

    return plot


def _azel_skyplot_cds_cols(
    azimuths: npt.NDArray[Float64_as_deg], elevations: npt.NDArray[Float64_as_deg]
):
    cols: dict[AzelSkyplotColumnKey, npt.NDArray] = {
        "azimuth": azimuths,
        "elevation": elevations,
        "adj_azimuth": azimuths - 90,
        "adj_elevation": 90 - elevations,
    }

    return cols


def azel_skyplot(azimuths: npt.NDArray[Float64_as_deg], elevations: npt.NDArray[Float64_as_deg]):
    """
    A plot similar to a skyplot.

    e.g. in the one in mathlab
    https://mathworks.com/help/satcom/ref/skyplot.html
    """

    cols = _azel_skyplot_cds_cols(azimuths, elevations)
    plot = _azel_skyplot_from_cds(source=bokeh_models.ColumnDataSource(t.cast(dict, cols)))

    return plot
