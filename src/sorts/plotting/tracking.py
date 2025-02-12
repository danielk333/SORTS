#!/usr/bin/env python

"""Tracking plot functions

"""

# Python standard import


# Third party import
import numpy as np
import matplotlib.pyplot as plt


# Local import
from .. import frames


def az_el_to_xy(az, el):
    az, el = np.radians(az), np.radians(el)

    r = np.cos(el)
    x = r * np.sin(az)
    y = r * np.cos(az)
    return x, y


def local_passes(passes, **kwargs):
    enu_ind = kwargs.pop("station_ind", 0)
    fig = None
    ax = None

    for ind, ps in enumerate(passes):
        if "add_track" not in kwargs:
            if ind == 0:
                kwargs["add_track"] = False
            else:
                kwargs["add_track"] = True

        if isinstance(ps.enu, list):
            enu = ps.enu[enu_ind]
        else:
            enu = ps.enu

        azelr = frames.cart_to_sph(enu[:3, :], degrees=kwargs.setdefault("degrees", True))

        _fig, ax = local_tracking(azelr[0, :], azelr[1, :], **kwargs)
        if fig is None:
            fig = _fig

        if "ax" not in kwargs:
            kwargs["ax"] = ax

    return fig, ax


def local_tracking(
    azimuth,
    elevation,
    ax=None,
    t=None,
    track_line_style="-",
    track_marker=None,
    track_color=None,
    add_track=False,
    node_times=False,
    degrees=True,
    alpha=1,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = None

    x0, y0 = az_el_to_xy(azimuth, elevation)
    ax.plot(x0, y0, c=track_color, ls=track_line_style, marker=track_marker, alpha=alpha)

    if not add_track:
        x, y = az_el_to_xy(np.linspace(0, 360, num=360), np.repeat(0.0, 360))
        ax.plot(x, y, color="green", alpha=0.5)

        x, y = az_el_to_xy(np.linspace(0, 360, num=360), np.repeat(30.0, 360))
        x[np.logical_and(x > 0, abs(y) < 0.1)] = np.nan
        ax.plot(x, y, color="black")
        ax.text(
            np.cos(np.pi * 30 / 180.0),
            0,
            r"$30\degree$",
            horizontalalignment="center",
            verticalalignment="center",
        )

        x, y = az_el_to_xy(np.linspace(0, 360, num=360), np.repeat(60.0, 360))
        x[np.logical_and(x > 0, abs(y) < 0.1)] = np.nan
        ax.plot(x, y, color="black")
        ax.text(
            np.cos(np.pi * 60 / 180.0),
            0,
            r"$60\degree$",
            horizontalalignment="center",
            verticalalignment="center",
        )

        x, y = az_el_to_xy(np.linspace(0, 360, num=360), np.repeat(80.0, 360))
        x[np.logical_and(x > 0, abs(y) < 0.1)] = np.nan
        ax.plot(x, y, color="black")
        ax.text(
            np.cos(np.pi * 80 / 180.0),
            0,
            r"$80\degree$",
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.plot(x0[0], y0[0], "o", alpha=alpha)
    ax.plot(x0[-1], y0[-1], "x", alpha=alpha)

    if t is None or not node_times:
        start = "Start"
        end = "End"
    else:
        start = t[0].to_value("isot", subfmt="date_hm")
        end = t[-1].to_value("isot", subfmt="date_hm")

    if node_times is not None:
        ax.text(x0[0], y0[0], start)
        ax.text(x0[-1], y0[-1], end)

    if not add_track:
        ax.text(0, 1, "North", horizontalalignment="center", verticalalignment="bottom")
        ax.text(0, -1, "South", horizontalalignment="center", verticalalignment="top")
        ax.text(1, 0, "East", horizontalalignment="left")
        ax.text(-1, 0, "West", horizontalalignment="right")

        ax.axis("off")
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect("equal", "datalim")

    return fig, ax
