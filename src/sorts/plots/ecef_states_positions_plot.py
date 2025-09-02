import typing as t
import numpy.typing as npt
import pandas as pd
import pyproj
import lets_plot as lp
import bokeh.plotting as bp
import bokeh.models as bokeh_models
from sorts.types import EcefStates
from sorts.utils import assert_class_attributes_equal_to
from sorts.frames import ITRS_to_geodetic

EcefStatesPositionsPlotColumnKey = t.Literal["lat", "lon", "wmx", "wmy"]


class _K_EcefStatesPositionsPlotColumnKey:
    """Internal helper class for accessing string keys consistently"""

    lat: t.Final = "lat"
    lon: t.Final = "lon"
    wmx: t.Final = "wmx"
    wmy: t.Final = "wmy"


assert_class_attributes_equal_to(
    _K_EcefStatesPositionsPlotColumnKey, t.get_args(EcefStatesPositionsPlotColumnKey)
)


def _ecef_states_positions_plot_from_cds(source: bokeh_models.ColumnarDataSource):
    """An internal ver of `ecef_states_positions_plot` that takes a bokeh `ColumnDataSource`."""

    _K = _K_EcefStatesPositionsPlotColumnKey
    plot = bp.figure(
        x_axis_type="mercator",
        y_axis_type="mercator",
        match_aspect=True,
    )
    plot.add_tile("CartoDB Positron", retina=True)

    scatter = plot.scatter(x=_K.wmx, y=_K.wmy, source=source)

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
                ("data (lat, lon)", f"(@{_K.lat}, @{_K.lon})"),
            ],
        )
    )

    return plot


def _ecef_states_positions_plot_cds_cols(ecefs: EcefStates):
    geodetic_coords = ITRS_to_geodetic(ecefs[0], ecefs[1], ecefs[2])
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857"
    )  # World Geodetic System to Web Mercator
    lat = geodetic_coords[0]
    lon = geodetic_coords[1]
    wmx, wmy = transformer.transform(lat, lon)

    cols: dict[EcefStatesPositionsPlotColumnKey, npt.NDArray] = {
        "lat": lat,
        "lon": lon,
        "wmx": wmx,
        "wmy": wmy,
    }

    return cols


# TODO: add down sampling? radar control slice are in milliseconds, while the simulation are in days or longer
def ecef_states_positions_plot(ecefs: EcefStates):
    cols = _ecef_states_positions_plot_cds_cols(ecefs)
    plot = _ecef_states_positions_plot_from_cds(
        source=bokeh_models.ColumnDataSource(t.cast(dict, cols))
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
