#!/usr/bin/env python

"""General plotting functions

"""

# Python standard import
from itertools import combinations

# Third party import
import numpy as np
import matplotlib.pyplot as plt

import pyorb

# Local import
from .. import constants


def kepler_orbit(orb, res=100, ax=None, **kwargs):
    """Plot a 3d kepler orbit from a single pyorb.Orbit item."""
    assert len(orb) == 1, "Can only handle one orbit for now"
    orbc = orb.copy()
    orbc.direct_update = False
    orbc.auto_update = False

    orbc.add(res - 1)
    kep0 = np.reshape(orb.kepler[:, 0], (6, 1))
    orbc.kepler = np.tile(kep0, (1, res))
    mu = np.linspace(0, 2 * np.pi, res)
    if orb.degrees:
        mu = np.degrees(mu)
    orbc.mean_anomaly = mu

    orbc.calculate_cartesian()
    states = orbc.cartesian

    if ax is None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = None
    ax.plot(states[0, :], states[1, :], states[2, :], "-", **kwargs)

    return fig, ax


def kepler_scatter(o, **options):
    """This function creates several scatter plots of a set of orbital elements based on the
    different possible axis planar projections, calculates all possible permutations of plane
    intersections based on the number of columns

    :param numpy.ndarray o: Rows are distinct orbits and columns are orbital elements in the order a, e, i, omega, Omega
    :param options:  dictionary containing all the optional settings

    #TODO: this needs updating

    Currently the options fields are:
        :marker [char]: the marker type
        :size [int]: the size of the marker
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :axis_labels [list of strings]: labels for each column
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :tight_rect [list of 4 floats]: configuration for the tight_layout function
        :usetex [bool]: whether to typeset labels using TeX syntax


    Example::

        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        orbs = np.matrix([
            11  + 3  *np.random.randn(1000),
            0.5 + 0.2*np.random.randn(1000),
            60  + 10 *np.random.randn(1000),
            120 + 5  *np.random.randn(1000),
            33  + 2  *np.random.randn(1000),
        ]).T

        dpt.orbits(orbs,
            title = "My orbital element distribution",
            size = 10,
        )


    """
    if not isinstance(o, np.ndarray):
        o = np.array(o)

    data_axis = options.get("axis", 0)
    if data_axis == 1:
        dim_axis = 0
    else:
        dim_axis = 1

    scale = options.get("scale", np.ones((o.shape[dim_axis],), dtype=o.dtype))

    # turn on TeX interpreter (or not)
    usetex = options["usetex"] if "usetex" in options else True
    plt.rc("text", usetex=usetex)

    lis = list(range(o.shape[dim_axis]))
    axis_plot = list(combinations(lis, 2))

    axis_labels = options.setdefault("axis_labels", None)

    limits = options.get("limits", None)

    if isinstance(axis_labels, str):
        if axis_labels == "earth-orbit":
            axis_labels = [
                "$a$ [$R_E$]",
                "$e$ [1]",
                "$i$ [deg]",
                "$\omega$ [deg]",
                "$\Omega$ [deg]",
                "$M_0$ [deg]",
            ]
            scale = [1 / constants.R_earth] + [1] * 5
        elif axis_labels == "earth-state":
            axis_labels = [
                "$x$ [$R_E$]",
                "$y$ [$R_E$]",
                "$z$ [$R_E$]",
                "$v_x$ [km/s]",
                "$v_y$ [km/s]",
                "$v_z$ [km/s]",
            ]
            scale = [1 / constants.R_earth] * 3 + [1e-3] * 3
        elif axis_labels == "sol-orbit":
            axis_labels = [
                "$a$ [AU]",
                "$e$ [1]",
                "$i$ [deg]",
                "$\omega$ [deg]",
                "$\Omega$ [deg]",
                "$M_0$ [deg]",
            ]
            scale = [1 / pyorb.AU] + [1] * 5
        elif axis_labels == "sol-state":
            axis_labels = [
                "$x$ [AU]",
                "$y$ [AU]",
                "$z$ [AU]",
                "$v_x$ [km/s]",
                "$v_y$ [km/s]",
                "$v_z$ [km/s]",
            ]
            scale = [1 / pyorb.AU] * 3 + [1e-3] * 3
        else:
            axis_labels = [""] * 6
    elif axis_labels is None:
        axis_labels = [""] * 6

    if o.shape[dim_axis] == 2:
        subplot_cnt = (1, 2)
        subplot_perms = 2
    elif o.shape[dim_axis] == 3:
        subplot_cnt = (1, 3)
        subplot_perms = 3
    elif o.shape[dim_axis] == 4:
        subplot_cnt = (2, 3)
        subplot_perms = 6
    elif o.shape[dim_axis] == 5:
        subplot_cnt = (2, 5)
        subplot_perms = 10
    else:
        subplot_cnt = (3, 5)
        subplot_perms = 15
    subplot_cnt_ind = 1

    if "window" in options:
        size_in = options["window"]
        size_in = tuple(x / 80.0 for x in size_in)
    else:
        size_in = (19, 10)

    fig = plt.figure(figsize=size_in, dpi=80)

    fig.suptitle(
        options.get("title", "Orbital elements distribution"),
        fontsize=options.get("title_font_size", 24),
    )
    axes = []
    for I in range(subplot_perms):
        ax = fig.add_subplot(subplot_cnt[0], subplot_cnt[1], subplot_cnt_ind)
        axes.append(ax)

        if dim_axis == 1:
            x = o[:, axis_plot[I][0]] * scale[axis_plot[I][0]]
            y = o[:, axis_plot[I][1]] * scale[axis_plot[I][1]]
        else:
            x = o[axis_plot[I][0], :] * scale[axis_plot[I][0]]
            y = o[axis_plot[I][1], :] * scale[axis_plot[I][1]]

        sc = ax.scatter(
            x.flatten(),
            y.flatten(),
            marker=options.get("marker", "."),
            c=options.setdefault("color", "b"),
            s=options.get("size", 2),
        )

        if isinstance(options["color"], np.ndarray):
            plt.colorbar(sc)

        # x_ticks = np.linspace(np.min(o[:,axis_plot[I][0]]),np.max(o[:,axis_plot[I][0]]), num=4)
        # plt.xticks( [round(x,1) for x in x_ticks] )
        ax.set_xlabel(
            axis_labels[axis_plot[I][0]],
            fontsize=options.setdefault("ax_font_size", 22),
        )
        ax.set_ylabel(
            axis_labels[axis_plot[I][1]],
            fontsize=options["ax_font_size"],
        )
        plt.xticks(fontsize=options.setdefault("tick_font_size", 17))
        plt.yticks(fontsize=options["tick_font_size"])

        if limits is not None:
            if len(limits) > axis_plot[I][0]:
                ax.set_xlim(*limits[axis_plot[I][0]])
            if len(limits) > axis_plot[I][1]:
                ax.set_ylim(*limits[axis_plot[I][1]])

        subplot_cnt_ind += 1

    plt.tight_layout(rect=options.setdefault("tight_rect", [0, 0.03, 1, 0.95]))

    return fig, axes
