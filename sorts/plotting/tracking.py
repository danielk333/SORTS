
def plot_angles(ts, angs, ax=None):
    '''Plot the angles data returned by the :func:`simulate_tracking.get_angles` function.

    :param list ts: List of times for each pass that the angles were evaluated over.
    :param list angs: List of angles for each pass.
    :param ax: matplotlib axis to plot the SNR's on. If not given, create new figure and axis.
    :return: The matplotlib axis object
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), tight_layout=True)
    txc = 0
    for txt,txa in zip(ts,angs):
        psc = 0
        for pst,psa in zip(txt,txa):
            tv = n.array(pst)/60.0
            ax.plot(tv - tv[0], n.array(psa),label='TX{} - pass {}'.format(txc,psc))
            psc+=1
        txc+=1
    ax.set( \
        title='passes Angles', \
        ylabel='Zenith angle [deg]', \
        xlabel='Time since entering FOV [min]')
    plt.legend()
    return ax


def plot_snr(t,all_snrs,radar, ax=None):
    '''Plots the SNR's structure (list of lists of numpy.ndarray's) returned by :func:`simulate_tracking.get_track_snr` and :func:`simulate_tracking.get_scan_snr`.

    :param numpy.ndarray t: Times corresponding to the evaluated SNR's.
    :param all_snrs: List structure returned by :func:`simulate_tracking.get_track_snr` and :func:`simulate_tracking.get_scan_snr`.
    :param RadarSystem radar: Radar system that measured the SNR's.
    :param ax: matplotlib axis to plot the SNR's on. If not given, create new figure and axis.
    :return: The matplotlib axis object
    '''
    
    tv = t/3600.0
    for txi,snrs in enumerate(all_snrs):
        if ax is None:
            fig, ax = plt.subplots(len(snrs), 1, figsize=(14, 10), tight_layout=True)
        for rxi,snr in enumerate(snrs):
            ax.plot(tv, 10.0*n.log10(snr), label='SNR: {} to {}'.format(radar._tx[txi].name,radar._rx[rxi].name))
        ax.set(
            ylabel='SNR [dB]',
            xlabel='time [h]',
        )
    plt.legend()
    return ax