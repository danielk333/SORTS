
def plot_radar_earth(ax, radar):
    for tx in radar._tx:
        ecef = coord.geodetic2ecef(tx.lat, tx.lon, tx.alt)
        ax.plot([ecef[0]],[ecef[1]],[ecef[2]],"x",color='r',label=tx.name)    
    for rx in radar._rx:
        ecef = coord.geodetic2ecef(rx.lat, rx.lon, rx.alt)
        ax.plot([ecef[0]],[ecef[1]],[ecef[2]],"x",color='b',label=rx.name)



def plot_radar_geo(radar):
    '''Plot the geographical location of the radar system using the GeoPandas library.

    To get:

    pip install git+git://github.com/geopandas/geopandas.git
    pip install descartes

    include in basic SORTS++ install?
    '''
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)

    df = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', linewidth=0.25, color='lightgrey', ax=ax)

    max_min_geo = {'lat': [0,0], 'lon': [0,0]}
    sites = []
    lats = []
    lons = []
    for r in radar._tx:
        lons.append(r.lon)
        lats.append(r.lat)
        sites.append(r.name)
    df_sites_tx = pd.DataFrame(
        {'Site': sites,
         'Latitude': lats,
         'Longitude': lons}
    )
    sites_rx = []
    lats_rx = []
    lons_rx = []
    for r in radar._rx:
        lons_rx.append(r.lon)
        lats_rx.append(r.lat)
        sites_rx.append(r.name)
    df_sites_rx = pd.DataFrame(
        {'Site': sites_rx,
         'Latitude': lats_rx,
         'Longitude': lons_rx}
    )
    
    max_min_geo['lat'][0] = np.min(np.array([lats + lats_rx]))
    max_min_geo['lat'][1] = np.max(np.array([lats + lats_rx]))
    max_min_geo['lon'][0] = np.min(np.array([lons + lons_rx]))
    max_min_geo['lon'][1] = np.max(np.array([lons + lons_rx]))
    _lat_df = 5.0
    _lon_df = 10.0

    if max_min_geo['lat'][0] > -(90.0 - _lat_df):
        max_min_geo['lat'][0] -= _lat_df
    if max_min_geo['lat'][1] < 90.0 - _lat_df:
        max_min_geo['lat'][1] += _lat_df

    if max_min_geo['lon'][0] > -(180.0 - _lon_df):
        max_min_geo['lon'][0] -= _lon_df
    if max_min_geo['lon'][1] < 180.0 - _lon_df:
        max_min_geo['lon'][1] += _lon_df

    _font = 8

    df_sites_rx['Coordinates'] = list(zip(df_sites_rx.Longitude, df_sites_rx.Latitude))
    df_sites_rx['Coordinates'] = df_sites_rx['Coordinates'].apply(Point)
    gdf_rx = geopandas.GeoDataFrame(df_sites_rx, geometry='Coordinates')
    gdf_rx.plot(ax=ax, color='blue', alpha=0.3, marker='h', markersize=24)
    for ind in range(gdf_rx.shape[0]):
        plt.text(gdf_rx['Coordinates'][ind].x+0.1, gdf_rx['Coordinates'][ind].y+0.2, sites_rx[ind], fontsize=_font)


    df_sites_tx['Coordinates'] = list(zip(df_sites_tx.Longitude, df_sites_tx.Latitude))
    df_sites_tx['Coordinates'] = df_sites_tx['Coordinates'].apply(Point)
    gdf = geopandas.GeoDataFrame(df_sites_tx, geometry='Coordinates')
    gdf.plot(ax=ax, color='red', alpha=0.7, marker='X', markersize=18)
    for ind in range(gdf.shape[0]):
        plt.text(gdf['Coordinates'][ind].x+0.1, gdf['Coordinates'][ind].y+0.5, sites[ind], fontsize=_font)

    ax.set_title('Geographical location for: {}'.format(radar.name))
    ax.set_xlim(*max_min_geo['lon'])
    ax.set_ylim(*max_min_geo['lat'])

    ax.set_aspect('equal')

    return fig, ax


def plot_radar(radar, save_folder = None):
    '''Plots aspects of the radar system.

    **Current plots:**

       * Geographical locations.
       * Antenna patterns.
       * Scan patterns.
    '''
    
    for tx in radar._tx:
        fig, ax = rs.plot_radar_scan(tx.scan, earth=True)
        if save_folder is not None:
            fig.savefig(save_folder + '/' + tx.name.replace(' ','_') + '_scan.png', bbox_inches='tight')
            plt.close(fig)

    for tx in radar._tx:
        fig, ax = antenna.plot_gain_heatmap(tx.beam, res=200, min_el = 75.0, title=tx.name)
        if save_folder is not None:
            fig.savefig(save_folder + '/' + tx.name.replace(' ','_') +'_'+ tx.beam.beam_name.replace(' ','_') + '.png', bbox_inches='tight')
            plt.close(fig)

    for rx in radar._rx:
        fig, ax = antenna.plot_gain_heatmap(rx.beam, res=200, min_el = 75.0, title=rx.name)
        if save_folder is not None:
            fig.savefig(save_folder + '/' + rx.name.replace(' ','_') +'_'+ rx.beam.beam_name.replace(' ','_') + '.png', bbox_inches='tight')
            plt.close(fig)

    fig, ax = plot_radar_geo(radar)
    if save_folder is not None:
        fig.savefig(save_folder + '/' + radar.name.replace(' ','_') + '.png', bbox_inches='tight')
        plt.close(fig)

    if save_folder is None:
        plt.show()

