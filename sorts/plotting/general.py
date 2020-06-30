#!/usr/bin/env python

'''Plotting helper functions

'''

#Python standard import
from itertools import combinations

#Third party import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#Local import
from .. import constants



def grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True):
    '''Add a 3d spherical grid to the given axis that represent the Earth.

    '''
    lons = np.linspace(-180, 180, num_lon+1) * np.pi/180 
    lons = lons[:-1]
    lats = np.linspace(-90, 90, num_lat) * np.pi/180 

    lonsl = np.linspace(-180, 180, res) * np.pi/180 
    latsl = np.linspace(-90, 90, res) * np.pi/180 

    r_e = constants.R_earth
    for lat in lats:
        x = r_e*np.cos(lonsl)*np.cos(lat)
        y = r_e*np.sin(lonsl)*np.cos(lat)
        z = r_e*np.ones(np.size(lonsl))*np.sin(lat)
        ax.plot(x,y,z,alpha=alpha,linestyle='-', marker='',color=color)

    for lon in lons:
        x = r_e*np.cos(lon)*np.cos(latsl)
        y = r_e*np.sin(lon)*np.cos(latsl)
        z = r_e*np.sin(latsl)
        ax.plot(x,y,z,alpha=alpha,color=color)
    
    if hide_ax:
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

    return ax


def hist2d(x, y, **options):
    """This function creates a histogram plot with lots of nice pre-configured settings unless they are overridden

    :param numpy.ndarray x:  data to histogram over, if x is not a vector it is flattened
    :param dict options: All keyword arguments as a dictionary containing all the optional settings.

    Currently the keyword arguments that can be used in the **options**:
        :bins [int]: the number of bins
        :colormap [str]: Name of colormap to use.
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :xlabel [string]: the label for the x axis
        :ylabel [string]: the label for the y axis
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :plot [tuple]: A tuple with the :code:`(fig, ax)` objects from matplotlib. Then no new figure and axis will be created.
        :logx [bool]: Determines if x-axis should be the logarithmic.
        :logy [bool]: Determines if y-axis should be the logarithmic.
        :log_freq [bool]: Determines if frequency should be the logarithmic.

    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        x = 10*np.random.randn(1000)

        dpt.hist(x,
           title = "My first plot",
        )
        

    """

    if not isinstance(x, np.ndarray):
        _x = np.array(x)
    else:
        _x = x.copy()

    if _x.size != _x.shape[0]:
        _x = _x.flatten()

    if options.setdefault('logx', False ):
        _x = np.log10(_x)

    if not isinstance(y, np.ndarray):
        _y = np.array(y)
    else:
        _y = y.copy()

    if _y.size != _y.shape[0]:
        _y = _y.flatten()

    if options.setdefault('logy', False ):
        _y = np.log10(_y)


    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(sz/80.0 for sz in size_in)
    else:
        size_in=(15, 7)
    if 'bin_size' in options:
        options['bins'] = int(np.round((np.max(_x)-np.min(_x))/options['bin_size']))

    if 'plot' in options:
        fig, ax = options['plot']
    else:
        fig, ax = plt.subplots(figsize=size_in,dpi=80)

    cmap = getattr(cm, options.setdefault('cmap', 'plasma'))

    if options.setdefault('log_freq', False ):
        hst = ax.hist2d(_x, _y,
            bins=options.setdefault('bins', int(np.round(np.sqrt(_x.size))) ),
            cmap=cmap,
            norm=mpl.colors.LogNorm(),
        )
        ax.set_facecolor(cmap.colors[0])
    else:
        hst = ax.hist2d(_x, _y,
            bins=options.setdefault('bins', int(np.round(np.sqrt(_x.size))) ),
            cmap=cmap,
            normed=options.setdefault('pdf', False),
        )
    title = options.setdefault('title','Data scatter plot')
    if title is not None:
        ax.set_title(title,
            fontsize=options.setdefault('title_font_size',24))
    ax.set_xlabel(options.setdefault('xlabel','X-axis'), 
        fontsize=options.setdefault('ax_font_size',20))
    ax.set_ylabel(options.setdefault('ylabel','Y-axis'),
            fontsize=options.setdefault('ax_font_size',20))
    cbar = plt.colorbar(hst[3], ax=ax)
    if options['log_freq']:
        cbar.set_label(options.setdefault('clabel','Logarithmic counts'),
            fontsize=options['ax_font_size'])
    else:
        cbar.set_label(options.setdefault('clabel','Counts'),
            fontsize=options['ax_font_size'])

    plt.xticks(fontsize=options.setdefault('tick_font_size',16))
    plt.yticks(fontsize=options['tick_font_size'])
    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, ax



def posterior(post, variables, **options):
    """This function creates several scatter plots of a set of orbital elements based on the
    different possible axis planar projections, calculates all possible permutations of plane
    intersections based on the number of columns

    :param numpy.ndarray post: Rows are distinct variable samples and columns are variables in the order of :code:`variables`
    :param list variables: Name of variables, used for axis names.
    :param options:  dictionary containing all the optional settings

    Currently the options fields are:
        :bins [int]: the number of bins
        :colormap [str]: Name of colormap to use.
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :axis_labels [list of strings]: labels for each column
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :tight_rect [list of 4 floats]: configuration for the tight_layout function


    """
    if type(post) != np.ndarray:
        post = np.array(post)

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    lis = list(range(post.shape[1]))
    axis_plot = list(combinations(lis, 2))

    axis_label = variables
    
    if post.shape[1] == 2:
        subplot_cnt = (1,2)
        subplot_perms = 2
    elif post.shape[1] == 3:
        subplot_cnt = (1,3)
        subplot_perms = 3
    elif post.shape[1] == 4:
        subplot_cnt = (2,3)
        subplot_perms = 6
    elif post.shape[1] == 5:
        subplot_cnt = (2,5)
        subplot_perms = 10
    else:
        subplot_cnt = (3,5)
        subplot_perms = 15
    subplot_cnt_ind = 1

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(x/80.0 for x in size_in)
    else:
        size_in=(19, 10)

    fig = plt.figure(figsize=size_in,dpi=80)

    cmap = options.setdefault('cmap', 'plasma')
    bins = options.setdefault('bins', int(np.round(np.sqrt(post.shape[0]))) )

    fig.suptitle(options.setdefault('title','Probability distribution'),
        fontsize=options.setdefault('title_font_size',24))
    axes = []
    for I in range( subplot_perms ):
        ax = fig.add_subplot(subplot_cnt[0],subplot_cnt[1],subplot_cnt_ind)
        axes.append(ax)
        x = post[:,axis_plot[I][0]]
        y = post[:,axis_plot[I][1]]
        fig, ax = hist2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            cmap=cmap,
            plot=(fig, ax),
            pdf=True,
            clabel='Probability',
            title = None,
            show=False,
        )
        ax.set_xlabel(axis_label[axis_plot[I][0]], 
            fontsize=options.setdefault('ax_font_size',22))
        ax.set_ylabel(axis_label[axis_plot[I][1]], 
            fontsize=options['ax_font_size'])
        plt.xticks(fontsize=options.setdefault('tick_font_size',17))
        plt.yticks(fontsize=options['tick_font_size'])
        subplot_cnt_ind += 1
    
    plt.tight_layout(rect=options.setdefault('tight_rect',[0, 0.03, 1, 0.95]))

    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, axes



def hist(x, **options):
    """This function creates a histogram plot with lots of nice pre-configured settings unless they are overridden

    :param numpy.ndarray x:  data to histogram over, if x is not a vector it is flattened
    :param dict options: All keyword arguments as a dictionary containing all the optional settings.

    Currently the keyword arguments that can be used in the **options**:
        :bins [int]: the number of bins
        :density [bool]: convert counts to density in [0,1]
        :edges [float]: bin edge line width, set to 0 to remove edges
        :title [string]: the title of the plot
        :title_font_size [int]: the title font size
        :xlabel [string]: the label for the x axis
        :ylabel [string]: the label for the y axis
        :tick_font_size [int]: the axis tick font size
        :window [tuple/list]: the size of the plot window in pixels (assuming dpi = 80)
        :save [string]: will not display figure and will instead save it to this path
        :show [bool]: if False will do draw() instead of show() allowing script to continue
        :plot [tuple]: A tuple with the :code:`(fig, ax)` objects from matplotlib. Then no new figure and axis will be created.
        :logx [bool]: Determines if x-axis should be the logarithmic.
        :logy [bool]: Determines if y-axis should be the logarithmic.

    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        x = 10*np.random.randn(1000)

        dpt.hist(x,
           title = "My first plot",
        )
        

    """

    if not isinstance(x, np.ndarray):
        _x = np.array(x)
    else:
        _x = x.copy()

    if _x.size != _x.shape[0]:
        _x = _x.flatten()

    if options.setdefault('logx', False ):
        _x = np.log10(_x)

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(sz/80.0 for sz in size_in)
    else:
        size_in=(15, 7)
    if 'bin_size' in options:
        options['bins'] = int(np.round((np.max(_x)-np.min(_x))/options['bin_size']))

    if 'plot' in options:
        fig, ax = options['plot']
    else:
        fig, ax = plt.subplots(figsize=size_in,dpi=80)

    ax.hist(_x,
        options.setdefault('bins', int(np.round(np.sqrt(_x.size))) ),
        density=options.setdefault('density',False),
        facecolor=options.setdefault('color','b'),
        edgecolor='black',
        linewidth=options.setdefault('edges',1.2),
        cumulative=options.setdefault('cumulative',False),
        log=options.setdefault('logy', False ),
        label=options.setdefault('label', None ),
        alpha=options.setdefault('alpha',1.0),
    )
    ax.set_title(options.setdefault('title','Data scatter plot'),\
        fontsize=options.setdefault('title_font_size',24))
    ax.set_xlabel(options.setdefault('xlabel','X-axis'), \
        fontsize=options.setdefault('ax_font_size',20))
    if options['density']:
        ax.set_ylabel(options.setdefault('ylabel','Density'), \
            fontsize=options['ax_font_size'])
    else:
        ax.set_ylabel(options.setdefault('ylabel','Frequency'), \
            fontsize=options['ax_font_size'])
    plt.xticks(fontsize=options.setdefault('tick_font_size',16))
    plt.yticks(fontsize=options['tick_font_size'])
    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, ax


def scatter(x, y, **options):
    """This function creates a scatter plot with lots of nice pre-configured settings unless they are overridden

    Currently the options fields are:
        :marker [char]: The marker type
        :size [int]: The size of the marker
        :alpha [float]: The transparency of the points.
        :title [string]: The title of the plot
        :title_font_size [int]: The title font size
        :xlabel [string]: The label for the x axis
        :ylabel [string]: The label for the y axis
        :tick_font_size [int]: The axis tick font size
        :window [tuple/list]: The size of the plot window in pixels (assuming dpi = 80)
        :save [string]: Will not display figure and will instead save it to this path
        :show [bool]: If False will do draw() instead of show() allowing script to continue
        :plot [tuple]: A tuple with the :code:`(fig, ax)` objects from matplotlib. Then no new figure and axis will be created.

    :param numpy.ndarray x:  x-axis data vector.
    :param numpy.ndarray y:  y-axis data vector.
    :param options:  dictionary containing all the optional settings.

    Example::
        
        import dpt_tools as dpt
        import numpy as np
        np.random.seed(19680221)

        x = 10*np.random.randn(100)
        y = 10*np.random.randn(100)

        dpt.scatter(x, y, {
            "title": "My first plot",
            } )

    """
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(x/80.0 for x in size_in)
    else:
        size_in=(15, 7)

    if 'plot' in options:
        fig, ax = options['plot']
    else:
        fig, ax = plt.subplots(figsize=size_in,dpi=80)

    ax.scatter([x], [y],
        marker=options.setdefault('marker','.'),
        c=options.setdefault('color','b'),
        s=options.setdefault('size',2),
        alpha=options.setdefault('alpha',0.75),
    )
    ax.set_title(options.setdefault('title','Data scatter plot'),
        fontsize=options.setdefault('title_font_size',24))
    ax.set_xlabel(options.setdefault('xlabel','X-axis'),
        fontsize=options.setdefault('ax_font_size',20))
    ax.set_ylabel(options.setdefault('ylabel','Y-axis'),
        fontsize=options['ax_font_size'])
    plt.xticks(fontsize=options.setdefault('tick_font_size',16))
    plt.yticks(fontsize=options['tick_font_size'])
    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, ax


def orbits(o, **options):
    """This function creates several scatter plots of a set of orbital elements based on the
    different possible axis planar projections, calculates all possible permutations of plane
    intersections based on the number of columns

    :param numpy.ndarray o: Rows are distinct orbits and columns are orbital elements in the order a, e, i, omega, Omega
    :param options:  dictionary containing all the optional settings

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
    if type(o) != np.ndarray:
        o = np.array(o)

    if 'unit' in options:
        if options['unit'] == 'km':
            o[:,0] = o[:,0]/6353.0
        elif options['unit'] == 'm':
            o[:,0] = o[:,0]/6353.0e3
    else:
        o[:,0] = o[:,0]/6353.0

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    lis = list(range(o.shape[1]))
    axis_plot = list(combinations(lis, 2))

    axis_label = options.setdefault('axis_labels', \
        [ "$a$ [$R_E$]","$e$ [1]","$i$ [deg]","$\omega$ [deg]","$\Omega$ [deg]","$M_0$ [deg]" ])
    
    if o.shape[1] == 2:
        subplot_cnt = (1,2)
        subplot_perms = 2
    elif o.shape[1] == 3:
        subplot_cnt = (1,3)
        subplot_perms = 3
    elif o.shape[1] == 4:
        subplot_cnt = (2,3)
        subplot_perms = 6
    elif o.shape[1] == 5:
        subplot_cnt = (2,5)
        subplot_perms = 10
    else:
        subplot_cnt = (3,5)
        subplot_perms = 15
    subplot_cnt_ind = 1

    if 'window' in options:
        size_in = options['window']
        size_in = tuple(x/80.0 for x in size_in)
    else:
        size_in=(19, 10)

    fig = plt.figure(figsize=size_in,dpi=80)

    fig.suptitle(options.setdefault('title','Orbital elements distribution'),\
        fontsize=options.setdefault('title_font_size',24))
    axes = []
    for I in range( subplot_perms ):
        ax = fig.add_subplot(subplot_cnt[0],subplot_cnt[1],subplot_cnt_ind)
        axes.append(ax)
        x = o[:,axis_plot[I][0]]
        y = o[:,axis_plot[I][1]]
        sc = ax.scatter( \
            x.flatten(), \
            y.flatten(), \
            marker=options.setdefault('marker','.'),\
            c=options.setdefault('color','b'),\
            s=options.setdefault('size',2))
        if isinstance(options['color'],np.ndarray):
            plt.colorbar(sc)
        x_ticks = np.linspace(np.min(o[:,axis_plot[I][0]]),np.max(o[:,axis_plot[I][0]]), num=4)
        plt.xticks( [round(x,1) for x in x_ticks] )
        ax.set_xlabel(axis_label[axis_plot[I][0]], \
            fontsize=options.setdefault('ax_font_size',22))
        ax.set_ylabel(axis_label[axis_plot[I][1]], \
            fontsize=options['ax_font_size'])
        plt.xticks(fontsize=options.setdefault('tick_font_size',17))
        plt.yticks(fontsize=options['tick_font_size'])
        subplot_cnt_ind += 1
    
    plt.tight_layout(rect=options.setdefault('tight_rect',[0, 0.03, 1, 0.95]))


    if 'unit' in options:
        if options['unit'] == 'km':
            o[:,0] = o[:,0]*6353.0
        elif options['unit'] == 'm':
            o[:,0] = o[:,0]*6353.0e3
    else:
        o[:,0] = o[:,0]*6353.0

    if 'save' in options:
        fig.savefig(options['save'],bbox_inches='tight')
    else:
        if options.setdefault('show', False):
            plt.show()
        else:
            plt.draw()

    return fig, axes
