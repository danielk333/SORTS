#!/usr/bin/env python

'''Radar configuration plot functions

'''

#Python standard import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

#Third party import


#Local import

from . import general
from ..transformations import frames


def radar_earth(ax, radar, **kwargs):

    tx_names = kwargs.pop('tx_names', None)
    rx_names = kwargs.pop('rx_names', None)
    general.grid_earth(ax, **kwargs)

    for ind, tx in enumerate(radar.tx):
        if tx_names is None:
            ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],"x",color='r')
        else:
            ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],"x",color='r',label=tx_names[ind])
        
    for ind, rx in enumerate(radar.rx):
        if rx_names is None:
            ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],"x",color='r')
        else:
            ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],"x",color='r',label=rx_names[ind])
    


def radar_map(radar, ax=None):
    '''Plot the geographical location of the radar system using the GeoPandas library.

    To get:

    pip install git+git://github.com/geopandas/geopandas.git
    pip install descartes

    '''

    try:
        import geopandas
    except ImportError:
        geopandas = None
        raise ImportError('Cannot plot geo-location of radar without "geopandas" package')

    if ax is None:
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)

    df = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', linewidth=0.25, color='lightgrey', ax=ax)

    max_min_geo = {'lat': [0,0], 'lon': [0,0]}
    sites = []
    lats = []
    lons = []
    for ind, r in enumerate(radar.tx):
        lons.append(r.lon)
        lats.append(r.lat)
        sites.append(f'TX-{ind}')
    df_sites_tx = pd.DataFrame(
        {'Site': sites,
         'Latitude': lats,
         'Longitude': lons}
    )
    sites_rx = []
    lats_rx = []
    lons_rx = []
    for ind, r in enumerate(radar.rx):
        lons_rx.append(r.lon)
        lats_rx.append(r.lat)
        sites_rx.append(f'RX-{ind}')
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
    df_sites_rx['Coordinates'] = df_sites_rx['Coordinates'].apply(frames.Point)
    gdf_rx = geopandas.GeoDataFrame(df_sites_rx, geometry='Coordinates')
    gdf_rx.plot(ax=ax, color='blue', alpha=0.3, marker='h', markersize=24)
    for ind in range(gdf_rx.shape[0]):
        plt.text(gdf_rx['Coordinates'][ind].x+0.1, gdf_rx['Coordinates'][ind].y+0.2, sites_rx[ind], fontsize=_font)


    df_sites_tx['Coordinates'] = list(zip(df_sites_tx.Longitude, df_sites_tx.Latitude))
    df_sites_tx['Coordinates'] = df_sites_tx['Coordinates'].apply(frames.Point)
    gdf = geopandas.GeoDataFrame(df_sites_tx, geometry='Coordinates')
    gdf.plot(ax=ax, color='red', alpha=0.7, marker='X', markersize=18)
    for ind in range(gdf.shape[0]):
        plt.text(gdf['Coordinates'][ind].x+0.1, gdf['Coordinates'][ind].y+0.5, sites[ind], fontsize=_font)

    ax.set_title('Geographical location for: {}'.format(radar.name))
    ax.set_xlim(*max_min_geo['lon'])
    ax.set_ylim(*max_min_geo['lat'])

    ax.set_aspect('equal')

    return ax

