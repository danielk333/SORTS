"""
Definition of colour schemes for lines and maps that also work for colour-blind
people. See https://personal.sron.nl/~pault/ for background information and
best usage of the schemes.

Copyright (c) 2021, Paul Tol
All rights reserved.

Modified by Daniel Kastinen 2022.
- Allow cycler settings
- Formats for inclusion in library, like raising errors instead of warnings
- Simplify constuction and more general cycles

License:  Standard 3-clause BSD
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array


CMAPS = {
    "sunset": (
        [
            "#364B9A",
            "#4A7BB7",
            "#6EA6CD",
            "#98CAE1",
            "#C2E4EF",
            "#EAECCC",
            "#FEDA8B",
            "#FDB366",
            "#F67E4B",
            "#DD3D2D",
            "#A50026",
        ],
        "#FFFFFF",
    ),
    "BuRd": (
        [
            "#2166AC",
            "#4393C3",
            "#92C5DE",
            "#D1E5F0",
            "#F7F7F7",
            "#FDDBC7",
            "#F4A582",
            "#D6604D",
            "#B2182B",
        ],
        "#FFEE99",
    ),
    "PRGn": (
        [
            "#762A83",
            "#9970AB",
            "#C2A5CF",
            "#E7D4E8",
            "#F7F7F7",
            "#D9F0D3",
            "#ACD39E",
            "#5AAE61",
            "#1B7837",
        ],
        "#FFEE99",
    ),
    "YlOrBr": (
        [
            "#FFFFE5",
            "#FFF7BC",
            "#FEE391",
            "#FEC44F",
            "#FB9A29",
            "#EC7014",
            "#CC4C02",
            "#993404",
            "#662506",
        ],
        "#888888",
    ),
    "WhOrBr": (
        [
            "#FFFFFF",
            "#FFF7BC",
            "#FEE391",
            "#FEC44F",
            "#FB9A29",
            "#EC7014",
            "#CC4C02",
            "#993404",
            "#662506",
        ],
        "#888888",
    ),
    "iridescent": (
        [
            "#FEFBE9",
            "#FCF7D5",
            "#F5F3C1",
            "#EAF0B5",
            "#DDECBF",
            "#D0E7CA",
            "#C2E3D2",
            "#B5DDD8",
            "#A8D8DC",
            "#9BD2E1",
            "#8DCBE4",
            "#81C4E7",
            "#7BBCE7",
            "#7EB2E4",
            "#88A5DD",
            "#9398D2",
            "#9B8AC4",
            "#9D7DB2",
            "#9A709E",
            "#906388",
            "#805770",
            "#684957",
            "#46353A",
        ],
        "#999999",
    ),
    "rainbow_PuRd": (
        [
            "#6F4C9B",
            "#6059A9",
            "#5568B8",
            "#4E79C5",
            "#4D8AC6",
            "#4E96BC",
            "#549EB3",
            "#59A5A9",
            "#60AB9E",
            "#69B190",
            "#77B77D",
            "#8CBC68",
            "#A6BE54",
            "#BEBC48",
            "#D1B541",
            "#DDAA3C",
            "#E49C39",
            "#E78C35",
            "#E67932",
            "#E4632D",
            "#DF4828",
            "#DA2222",
        ],
        "#FFFFFF",
    ),
    "rainbow_PuBr": (
        [
            "#6F4C9B",
            "#6059A9",
            "#5568B8",
            "#4E79C5",
            "#4D8AC6",
            "#4E96BC",
            "#549EB3",
            "#59A5A9",
            "#60AB9E",
            "#69B190",
            "#77B77D",
            "#8CBC68",
            "#A6BE54",
            "#BEBC48",
            "#D1B541",
            "#DDAA3C",
            "#E49C39",
            "#E78C35",
            "#E67932",
            "#E4632D",
            "#DF4828",
            "#DA2222",
            "#B8221E",
            "#95211B",
            "#721E17",
            "#521A13",
        ],
        "#FFFFFF",
    ),
    "rainbow_WhRd": (
        [
            "#E8ECFB",
            "#DDD8EF",
            "#D1C1E1",
            "#C3A8D1",
            "#B58FC2",
            "#A778B4",
            "#9B62A7",
            "#8C4E99",
            "#6F4C9B",
            "#6059A9",
            "#5568B8",
            "#4E79C5",
            "#4D8AC6",
            "#4E96BC",
            "#549EB3",
            "#59A5A9",
            "#60AB9E",
            "#69B190",
            "#77B77D",
            "#8CBC68",
            "#A6BE54",
            "#BEBC48",
            "#D1B541",
            "#DDAA3C",
            "#E49C39",
            "#E78C35",
            "#E67932",
            "#E4632D",
            "#DF4828",
            "#DA2222",
        ],
        "#666666",
    ),
    "rainbow_WhBr": (
        [
            "#E8ECFB",
            "#DDD8EF",
            "#D1C1E1",
            "#C3A8D1",
            "#B58FC2",
            "#A778B4",
            "#9B62A7",
            "#8C4E99",
            "#6F4C9B",
            "#6059A9",
            "#5568B8",
            "#4E79C5",
            "#4D8AC6",
            "#4E96BC",
            "#549EB3",
            "#59A5A9",
            "#60AB9E",
            "#69B190",
            "#77B77D",
            "#8CBC68",
            "#A6BE54",
            "#BEBC48",
            "#D1B541",
            "#DDAA3C",
            "#E49C39",
            "#E78C35",
            "#E67932",
            "#E4632D",
            "#DF4828",
            "#DA2222",
            "#B8221E",
            "#95211B",
            "#721E17",
            "#521A13",
        ],
        "#666666",
    ),
}

CMAPS_DISCRETE = {
    "sunset_discrete": (
        [
            "#364B9A",
            "#4A7BB7",
            "#6EA6CD",
            "#98CAE1",
            "#C2E4EF",
            "#EAECCC",
            "#FEDA8B",
            "#FDB366",
            "#F67E4B",
            "#DD3D2D",
            "#A50026",
        ],
        "#FFFFFF",
    ),
    "BuRd_discrete": (
        [
            "#2166AC",
            "#4393C3",
            "#92C5DE",
            "#D1E5F0",
            "#F7F7F7",
            "#FDDBC7",
            "#F4A582",
            "#D6604D",
            "#B2182B",
        ],
        "#FFEE99",
    ),
    "PRGn_discrete": (
        [
            "#762A83",
            "#9970AB",
            "#C2A5CF",
            "#E7D4E8",
            "#F7F7F7",
            "#D9F0D3",
            "#ACD39E",
            "#5AAE61",
            "#1B7837",
        ],
        "#FFEE99",
    ),
    "YlOrBr_discrete": (
        [
            "#FFFFE5",
            "#FFF7BC",
            "#FEE391",
            "#FEC44F",
            "#FB9A29",
            "#EC7014",
            "#CC4C02",
            "#993404",
            "#662506",
        ],
        "#888888",
    ),
}


def rainbow_discrete(lut=None):
    clrs = [
        "#E8ECFB",
        "#D9CCE3",
        "#D1BBD7",
        "#CAACCB",
        "#BA8DB4",
        "#AE76A3",
        "#AA6F9E",
        "#994F88",
        "#882E72",
        "#1965B0",
        "#437DBF",
        "#5289C7",
        "#6195CF",
        "#7BAFDE",
        "#4EB265",
        "#90C987",
        "#CAE0AB",
        "#F7F056",
        "#F7CB45",
        "#F6C141",
        "#F4A736",
        "#F1932D",
        "#EE8026",
        "#E8601C",
        "#E65518",
        "#DC050C",
        "#A5170E",
        "#72190E",
        "#42150A",
    ]
    indexes = [
        [9],
        [9, 25],
        [9, 17, 25],
        [9, 14, 17, 25],
        [9, 13, 14, 17, 25],
        [9, 13, 14, 16, 17, 25],
        [8, 9, 13, 14, 16, 17, 25],
        [8, 9, 13, 14, 16, 17, 22, 25],
        [8, 9, 13, 14, 16, 17, 22, 25, 27],
        [8, 9, 13, 14, 16, 17, 20, 23, 25, 27],
        [8, 9, 11, 13, 14, 16, 17, 20, 23, 25, 27],
        [2, 5, 8, 9, 11, 13, 14, 16, 17, 20, 23, 25],
        [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 20, 23, 25],
        [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25],
        [2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
        [2, 4, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
        [2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27],
        [2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26, 27],
        [1, 3, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26, 27],
        [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26, 27],
        [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27],
        [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27, 28],
        [0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27, 28],
    ]
    if lut is None or lut < 1 or lut > 23:
        lut = 22
    cmap = discretemap("rainbow_discrete", [clrs[i] for i in indexes[lut - 1]])
    if lut == 23:
        cmap.set_bad("#777777")
    else:
        cmap.set_bad("#FFFFFF")

    return cmap


CMAP_SPECIAL = {
    "rainbow_discrete": rainbow_discrete,
}

CSET = {
    "bright": (
        "blue red green yellow cyan purple grey black",
        ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"],
    ),
    "high-contrast": (
        "blue yellow red black",
        ["#004488", "#DDAA33", "#BB5566", "#000000"],
    ),
    "vibrant": (
        "orange blue cyan magenta red teal grey black",
        ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB", "#000000"],
    ),
    "muted": (
        "rose indigo sand green cyan wine teal olive purple pale_grey black",
        [
            "#CC6677",
            "#332288",
            "#DDCC77",
            "#117733",
            "#88CCEE",
            "#882255",
            "#44AA99",
            "#999933",
            "#AA4499",
            "#DDDDDD",
            "#000000",
        ],
    ),
    "medium-contrast": (
        "light_blue dark_blue light_yellow dark_red dark_yellow light_red black",
        ["#6699CC", "#004488", "#EECC66", "#994455", "#997700", "#EE99AA", "#000000"],
    ),
    "light": (
        "light_blue orange light_yellow pink light_cyan mint pear olive pale_grey black",
        [
            "#77AADD",
            "#EE8866",
            "#EEDD88",
            "#FFAABB",
            "#99DDFF",
            "#44BB99",
            "#BBCC33",
            "#AAAA00",
            "#DDDDDD",
            "#000000",
        ],
    ),
}

COLORMAPS = {
    "cmap": list(CMAPS.keys()),
    "cmap_discrete": list(CMAPS_DISCRETE.keys()),
    "cmap_special": list(CMAP_SPECIAL.keys()),
    "cset": list(CSET.keys()),
}


def discretemap(colormap, hexclrs):
    """
    Produce a colormap from a list of discrete colors without interpolation.
    """
    clrs = to_rgba_array(hexclrs)
    clrs = np.vstack([clrs[0], clrs, clrs[-1]])
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [
            (i / (len(clrs) - 2.0), clrs[i, ki], clrs[i + 1, ki]) for i in range(len(clrs) - 1)
        ]
    return LinearSegmentedColormap(colormap, cdict)


def get_cmap(colormap=None, **kwargs):
    """
    Continuous and discrete color sets for ordered data.

    Returns a matplotlib colormap.
    """
    if colormap is None:
        return list(CMAPS.keys()) + list(CMAPS_DISCRETE.keys()) + list(CMAP_SPECIAL.keys())

    if colormap in CMAPS:
        clrs, bad = CMAPS[colormap]
        cmap = LinearSegmentedColormap.from_list(colormap, clrs)
        cmap.set_bad(bad)
    elif colormap in CMAPS_DISCRETE:
        clrs, bad = CMAPS[colormap]
        cmap = discretemap(colormap, clrs)
        cmap.set_bad(bad)
    elif colormap in CMAP_SPECIAL:
        func = CMAP_SPECIAL[colormap]
        cmap = func(**kwargs)
    else:
        raise ValueError(
            "*** Warning: requested colormap not defined," + f"known colormaps are {COLORMAPS}."
        )

    return cmap


def get_cset(colorset=None):
    """
    Discrete color sets for qualitative data.

    Define a namedtuple instance with the colors.
    Examples for: cset = tol_cset(<scheme>)
      - cset.red and cset[1] give the same color (in default 'bright' colorset)
      - cset._fields gives a tuple with all color names
      - list(cset) gives a list with all colors
    """

    if colorset is None:
        return list(CSET.keys())

    if colorset in CSET:
        col_names, cols = CSET[colorset]
        ctup = namedtuple(colorset, col_names)
        cset = ctup(*cols)
    else:
        raise ValueError(
            "*** Warning: requested colorset not defined,"
            + "known colorsets are {}.".format(list(CSET.keys()))
        )

    return cset


def get_cycle(name):
    if name in CSET:
        clrs = list(get_cset(name))
    elif name in CMAPS:
        clrs, bad = CMAPS[name]
    elif name in CMAPS_DISCRETE:
        clrs, bad = CMAPS_DISCRETE[name]

    return plt.cycler("color", clrs)


def set_cycle(ax, name="vibrant"):
    ax.set_prop_cycle(get_cycle(name))
