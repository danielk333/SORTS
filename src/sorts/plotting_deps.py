import logging

logger = logging.getLogger(__name__)

has_plotting_deps = False

try:
    import geopandas as gpd
    import geodatasets
    import lets_plot as lp
    import lets_plot.geo_data as lp_geo_data
    import bokeh.plotting as bp
    import bokeh.models as bokeh_models

    lp.LetsPlot.setup_html()

    has_plotting_deps = True

except ImportError as err:
    logger.warning(
        "Plotting depenencies not found, the functionalities are therefore not available."
    )
