#!/usr/bin/env python

'''Errors

'''
#Python standard import
import pathlib
import pkg_resources

#Third party import
import numpy as np
import h5py
import scipy.interpolate
import scipy.constants
import scipy.signal
from tqdm import tqdm

#Local import
from .errors import Errors

def simulate_echo(codes, t_vecs, bw=1e6, dop_Hz=0.0, range_m=1e3, sr=5e6):
    ''' Simulate a radar echo with range and Doppler.
    
    This function uses windowing to simulate a continuous finite bandwidth signal.
    This is used for linearized error estimates of range and range-rate errors.
    This function assumes a mono-static radar configuration. 

    Parameters
    ----------
    codes : numpy.ndarray (M, N)
        Sequence of encoded signals. 

        >>>  codes : [[Pulse 1], [Pulse 2], .... , [Pulse 3]]
    t_vecs : numpy.ndarray (M,)
        Transmission start time (in seconds).
    bw : float, default=1e6
        Transmitted signal bandwidth (in Hertz).
    dop_Hz : float, default=0.0
        Echo Doppler shift due to the target velocity (in Hertz). 
    range_m : float, default=1e3
        Target range (in meters).
    sr : float, default=5e6
        Sampling rate (in Hertz).

    Returns
    -------
    z : numpy.ndarray of complex
        Echo of the coded radar signal 
    '''
    # init computations
    codelen = len(codes[0])
    n_codes = len(codes)
    tvec = np.zeros(codelen*n_codes)
    z = np.zeros(codelen*n_codes,dtype=np.complex64)

    # compute signal envelope
    for ci, code in enumerate(codes):
        z[np.arange(codelen) + ci*codelen] = code
        tvec[np.arange(codelen) + ci*codelen] = t_vecs[ci]

    # setup envelop interpolation function
    tvec_i = np.copy(tvec)
    tvec_i[0] = tvec[0] - 1e99
    tvec_i[len(tvec)-1] = tvec[len(tvec)-1] + 1e99
    zfun = scipy.interpolate.interp1d(tvec_i, z, kind="linear")

    # compute time shift due to range
    dt = 2.0*range_m/scipy.constants.c

    # compute radar echo
    z = zfun(tvec+dt)*np.exp(1j*np.pi*2.0*dop_Hz*tvec)

    return z


def lin_error(enr=10.0, txlen=1000.0, n_ipp=10, ipp=20e-3, bw=1e6, dr=10.0, ddop=1.0, sr=100e6):
    ''' Determines linearized errors for range and range-rate error
    for a psuedorandom binary phase coded radar transmit pulse
    with a certain transmit bandwidth (inverse of bit length).

    This function calculates the line of sight range and range-rate error,
    given the ENR after coherent integration (pulse compression) and the
    pulse length ``txlen`` in microseconds.

    Finally, this function simulates a measurement and performs a linearized error estimate.
    
    Parameters
    ----------  
    enr : float, default=10.0
        Signal Energy-to-Noise ratio.
    txlen : float, default=1000.0
        Transmitted pulse length (microseconds).
    n_ipp : int, default=10
        Number of pulses.
    ipp : float, default=20e-3
        Inter-Pulse Period (in seconds).
    bw : float, default=1e6
        Transmitted signal bandwidth (in hertz).
    dr : float, default=10.0
        Range step used for linearization (in meters).
    ddop : float, default=1.0
        Doppler frequency step used for linearization (in hertz).
    sr : float, default=100e6
        Sampling rate (in hertz).

    Returns
    -------
    S : numpy.ndarray (2,)
        Linearized error standard deviation (*range* and *range rate*).
    '''
    # setup computations
    codes = []
    t_vecs = []    
    n_bits = int(bw*txlen/1e6)
    oversample = int(sr/bw)
    wfun = scipy.signal.hamming(oversample)
    wfun = wfun/np.sum(wfun)

    # compute transmitted signal properties and enveloppe
    for i in range(n_ipp):
        bits = np.array(np.sign(np.random.randn(n_bits)),dtype=np.complex64)
        zcode = np.zeros(n_bits*oversample+2*oversample,dtype=np.complex64)
        for j in range(oversample):
            zcode[np.arange(n_bits)*oversample + j + oversample] = bits

        # filter signal so that phase transitions are not too sharp
        zcode = np.convolve(wfun, zcode, mode="same")
        codes.append(zcode)
        tcode = np.arange(n_bits*oversample + 2*oversample)/sr + float(i)*ipp
        t_vecs.append(tcode)

    # compute radar echo at the target
    z0 = simulate_echo(codes, t_vecs, dop_Hz=0.0, range_m=0.0, bw=bw, sr=sr)
    tau = float(n_ipp)*txlen/1e6
    
    # convert coherently integrated ENR to SNR (variance of the measurement errors at receiver bandwidth)
    snr = enr/(tau*sr)

    # compute signal partial derivatives (w.r.t. range and range rates) 
    z_dr = simulate_echo(codes, t_vecs, dop_Hz=0.0, range_m=dr, bw=bw, sr=sr)
    z_diff_r = (z0 - z_dr)/dr    
    z_ddop = simulate_echo(codes, t_vecs, dop_Hz=ddop, range_m=0.0, bw=bw, sr=sr)
    z_diff_dop = (z0 - z_ddop)/ddop
    
    # compute linearized error covariance matrix
    t_l = len(z_dr)
    A = np.zeros([t_l,2],dtype=np.complex64)
    A[:,0] = z_diff_r
    A[:,1] = z_diff_dop
    S = np.real(np.linalg.inv(np.dot(np.transpose(np.conj(A)),A))/snr)

    return np.sqrt(np.diag(S))


def precalculate_dr(txlen, bw, ipp=20e-3, n_ipp=20, n_interp=20):
    ''' Computes **range** and **range rate** linearized errors to perform an 
    interpolation over a set of Energy-to-Noise Ratios.

    This function computes the linearized errors due to the propagation of 
    coded signals. It returns those errors over a set of points optimized 
    for interpolation.

    Parameters
    ----------
    txlen : float, default=1000.0
        Transmitted pulse length (microseconds).
    bw : float, default=1e6
        Transmitted signal bandwidth (in hertz).
    ipp : float, default=20e-3
        Inter-Pulse Period (in seconds).
    n_ipp : int, default=10
        Number of pulses.
    n_interp : int, default=20
        Number of ENR values used to perform the computations.

    Returns
    -------
    enrs
        Energy-to-Noise Ratio.
    drs
        Range errors (in Hertz).
    ddops
        Doppler shift (range-rate) errors (in Hertz).
    '''
    # set up computations
    enrs = 10.0**np.linspace(-3, 10, num=n_interp)
    drs = np.zeros(n_interp)
    ddops = np.zeros(n_interp)  

    # compute each range and Doppler errors.
    q = tqdm(total=n_interp)
    for ei, s in enumerate(enrs):
        dr, ddop = lin_error(enr=s, txlen=txlen, bw=bw, ipp=ipp, n_ipp=n_ipp)
        drs[ei] = dr
        ddops[ei] = ddop
        q.update(1)
        q.set_description(f'Linear errors at SNR {10*np.log10(s)} dB')
    q.close()

    return enrs, drs, ddops


class LinearizedCoded(Errors):
    ''' Standard encoded signal errors computation class.
    
    The :class:`LinearizedCoded` provides a standard interface
    to compute linearized errors of encoded signals. More specifically,
    it allows to compute the **range** and **range-rate** errors as a function
    of the signal properties and signal to noise ratio by performing
    an interpolation over a set of precalculated linearized errors.

    Parameters 
    ----------
    tx : :class:`sorts.station.TX<sorts.radar.system.station.TX>`
        Transmitting station.
    seed : int=None
        Seed used to generate random number using numpy. 
    cache_folder : str or :class:`pathlib.Path`, default=None
        Path of the folder where computations results of linearised errors are cached.
    min_range_rate_std : float, default=0.1
        Minimum range rate dopple measurement standard deviation (in hertz).
    min_range_std : float, default=0.1
        Minimum range measurement standard deviation (in meters).

    Example
    -------
    This simple example showcases the use of the :class:`LinearizedCoded` class to compute the perturbations 
    in range measurements due to ``Signal-to-Noise Ratio`` fluctuations. 

    .. code-block:: Python

        import sorts
        import numpy as np
        import matplotlib.pyplot as plt

        radar = sorts.radars.eiscat3d

        # initialization of the linearized errors for coded signals
        err = sorts.measurement_errors.LinearizedCoded(radar.tx[0], seed=123)

        # number of ranges
        num = 1000
        # number of range bins for posterior distribution estimate
        n_bins = 50

        # generate 100 range values and 
        ranges = np.linspace(300e3, 350e3, num=num)[::-1]

        # generate random SNR values
        snrs = np.random.randn(num)*15.0 + 20**1.0
        snrs[snrs<0.1] = 0.1

        # compute perturbated range estimates due to SNR fluctuations
        perturbed_ranges = err.range(ranges, snrs)

        # plot results
        fig, axes = plt.subplots(3, 1)
        axes[0].plot(np.arange(0, num), ranges, "--k") # initial range values
        axes[0].plot(np.arange(0, num), perturbed_ranges, "-r") # perturbed range values
        axes[0].set_xlabel("$N$ [$-$]")
        axes[0].set_ylabel("$r$ [$m$]")

        axes[1].hist(10*np.log10(snrs), n_bins, color="blue") # snr distribution
        axes[1].set_ylabel("$N$ [$-$]")
        axes[1].set_xlabel("$SNR$ [$dB$]")

        axes[2].hist(ranges - perturbed_ranges, n_bins, color="blue") # range error distribution 
        axes[2].set_ylabel("$N$ [$-$]")
        axes[2].set_xlabel("$r-r_{est}$ [$m$]")
        plt.show()

    .. rubric:: results

    .. figure:: ../../../../figures/errors_example_ranges.png
    '''
    VARIABLES = [
        'range', 
        'doppler',
        'range_rate',
    ]
    ''' Random variables used for error estimation. ''' 

    def __init__(self, tx, seed=None, cache_folder=None, min_range_rate_std=0.1, min_range_std=0.1):
        super().__init__(seed=seed)

        self.min_range_std = min_range_std
        ''' Random variables used for error estimation. ''' 
        self.min_range_rate_std = min_range_rate_std
        ''' Random variables used for error estimation. ''' 

        bw = tx.bandwidth
        txlen = tx.pulse_length*1e6 # pulse length in microseconds
        ipp = tx.ipp
        n_ipp = int(tx.n_ipp)
        self.freq = tx.beam.frequency

        fname = f'MCerr_n{n_ipp}_tx{int(txlen)}_ipp{int(ipp*1e6)}_bw{int(bw)}.h5'
        if cache_folder is None:
            try:
                stream = pkg_resources.resource_stream('sorts.data', fname)
                h = h5py.File(stream,'r')
                enrs = np.copy(h['enrs'][()])
                drs = np.copy(h['drs'][()])
                ddops = np.copy(h['ddops'][()])
                h.close()
            except:
                # compute range/doppler errors by performing a direct monte carlo simulatio
                enrs, drs, ddops = precalculate_dr(txlen, bw, ipp=ipp, n_ipp=n_ipp, n_interp=20)
        else:
            if not isinstance(cache_folder, pathlib.Path):
                cache_folder = pathlib.Path(cache_folder)
            pth = cache_folder / fname

            if pth.is_file():
                h = h5py.File(pth,'r')
                enrs = np.copy(h['enrs'][()])
                drs = np.copy(h['drs'][()])
                ddops = np.copy(h['ddops'][()])
                h.close()
            else:
                enrs, drs, ddops = precalculate_dr(txlen, bw, ipp=ipp, n_ipp=n_ipp, n_interp=20)
                ho = h5py.File(pth,"w")
                ho["enrs"] = enrs
                ho["drs"] = drs
                ho["ddops"] = ddops
                ho.close()

        self.rfun = scipy.interpolate.interp1d(np.log10(enrs), np.log10(drs))
        ''' Linearized range error function ``error = f(enrs[dB])``. ''' 
        self.dopfun = scipy.interpolate.interp1d(np.log10(enrs), np.log10(ddops))
        ''' Linearized range-rate Doppler error function ``error = f(enrs[dB])``. ''' 

    def range(self, data, snr):
        ''' Adds random gaussian errors to range data in meters. '''
        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*self.range_std(snr)


    def doppler(self, data, snr):
        ''' Adds random gaussian errors to Doppler shift data in hertz. '''
        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*self.doppler_std(snr)


    def range_rate(self, data, snr):
        ''' Adds random gaussian errors to range-rate data in m/s. '''
        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*self.range_rate_std(snr)


    def range_std(self, snr):
        ''' Computes the expected standard deviation of range measurements in meters. '''
        dr = 10.0**(self.rfun(np.log10(snr)))

        # if object diameter is larger than range error, make it at least as big as target
        if self.min_range_std is not None:
            if len(dr.shape) == 0:
                if dr < self.min_range_std:
                    dr = self.min_range_std
            else:
                dr[dr < self.min_range_std] = self.min_range_std
        return dr


    def doppler_std(self, snr):
        ''' Computes the expected standard deviation of Doppler measurements in Hertz. '''
        ddop = 10.0**(self.dopfun(np.log10(snr)))


        if len(ddop.shape) == 0:
            if ddop < self.min_range_rate_std:
                ddop = self.min_range_rate_std
        else:
            ddop[ddop < self.min_range_rate_std] = self.min_range_rate_std
        return ddop


    def range_rate_std(self, snr):
        ''' Computes the expected standard deviation of range-rate measurements in m/s. '''
        rr_std = scipy.constants.c*self.doppler_std(snr)/self.freq/2.0
        return rr_std



class LinearizedCodedIonospheric(LinearizedCoded):
    ''' Standard encoded signal errors computation class (with Ionospheric errors).
    
    The :class:`LinearizedCoded` provides a standard interface
    to compute linearized errors of encoded signals. More specifically,
    it allows to compute the **range** and **range-rate** errors as a function
    of the signal properties and signal to noise ratio by performing
    an interpolation over a set of precalculated linearized errors.

    This class also estimates the range error due to signal propagation in the ionosphere.
    Those errors are estimated by performing an interpolation within a table 
    of error values estimated at f=233MHz whith the radar pointing in the zenith direction
    for different target ranges.

    Parameters 
    ----------
    tx : :class:`sorts.station.TX<sorts.radar.system.station.TX>`
        Transmitting station.
    seed : int=None
        Seed used to generate random number using numpy. 
    cache_folder : str or :class:`pathlib.Path`, default=None
        Path of the folder where computations results of linearised errors are cached.
    min_range_rate_std : float, default=0.1
        Minimum range rate dopple measurement standard deviation (in hertz).
    min_range_std : float, default=0.1
        Minimum range measurement standard deviation (in meters).
    ''' 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ionospheric range errors @ 233MHz (zenith)
        r = np.array([0.0, 1000.0, 1000000.0])
        e = np.array([0.0, 0.1, 100.0])*150.0

        self.iono_errfun = scipy.interpolate.interp1d(r,e)
        ''' Ionospheric range error function ``error = f(range)``. '''


    def range_std(self, path_range, snr):
        ''' Computes the expected standard deviation of range measurements in meters. '''
        dr = 10.0**(self.rfun(np.log10(snr)))

        # add ionospheric error
        dr = np.sqrt(dr**2.0 + self.iono_errfun(path_range/1e3)**2.0)

        if self.min_range_std is not None:
            if len(dr.shape) == 0:
                if dr < self.min_range_std:
                    dr = self.min_range_std
            else:
                dr[dr < self.min_range_std] = self.min_range_std

        return dr


    def range(self, data, snr):
        ''' Adds random gaussian errors to range data in meters. '''
        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*self.range_std(data, snr)

