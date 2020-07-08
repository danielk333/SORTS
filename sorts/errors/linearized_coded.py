#!/usr/bin/env python

'''Errors

'''
#Python standard import
import pathlib
import importlib.resources

#Third party import
import numpy as np
import h5py
import scipy.interpolate
import scipy.constants
import scipy.signal
from tqdm import tqdm

#Local import
from .errors import Errors

def simulate_echo(codes,t_vecs,bw=1e6,dop_Hz=0.0,range_m=1e3,sr=5e6):
    '''
    Simulate a radar echo with range and doppler.
    Use windowing to simulate a continuous finite bandwidth signal.
    This is used for linearized error estimates of range and range-rate errors.
    '''

    codelen=len(codes[0])
    n_codes=len(codes)
    tvec=np.zeros(codelen*n_codes)
    z=np.zeros(codelen*n_codes,dtype=np.complex64)
    for ci,code in enumerate(codes):
        z[np.arange(codelen)+ci*codelen]=code
        tvec[np.arange(codelen)+ci*codelen]=t_vecs[ci]
    tvec_i=np.copy(tvec)
    tvec_i[0]=tvec[0]-1e99
    tvec_i[len(tvec)-1]=tvec[len(tvec)-1]+1e99        
    zfun=scipy.interpolate.interp1d(tvec_i,z,kind="linear")
    dt = 2.0*range_m/scipy.constants.c

    z=zfun(tvec+dt)*np.exp(1j*np.pi*2.0*dop_Hz*tvec)

    return(z)

def lin_error(enr=10.0,txlen=1000.0,n_ipp=10,ipp=20e-3,bw=1e6,dr=10.0,ddop=1.0,sr=100e6):
    '''
     Determine linearized errors for range and range-rate error
     for a psuedorandom binary phase coded radar transmit pulse
     with a certain transmit bandwidth (inverse of bit length)

     calculate line of sight range and range-rate error,
     given ENR after coherent integration (pulse compression)
     txlen in microseconds.

    Simulate a measurement and do a linearized error estimate.

    '''
    codes=[]
    t_vecs=[]    
    n_bits = int(bw*txlen/1e6)
    oversample=int(sr/bw)
    wfun=scipy.signal.hamming(oversample)
    wfun=wfun/np.sum(wfun)
    for i in range(n_ipp):
        bits=np.array(np.sign(np.random.randn(n_bits)),dtype=np.complex64)
        zcode=np.zeros(n_bits*oversample+2*oversample,dtype=np.complex64)
        for j in range(oversample):
            zcode[np.arange(n_bits)*oversample+j+oversample]=bits
        # filter signal so that phase transitions are not too sharp
        zcode=np.convolve(wfun,zcode,mode="same")
        codes.append(zcode)
        tcode=np.arange(n_bits*oversample+2*oversample)/sr + float(i)*ipp
        t_vecs.append(tcode)

    z0=simulate_echo(codes,t_vecs,dop_Hz=0.0,range_m=0.0,bw=bw,sr=sr)
    tau=(float(n_ipp)*txlen/1e6)
    
    # convert coherently integrated ENR to SNR (variance of the measurement errors at receiver bandwidth)
    snr = enr/(tau*sr)

    z_dr=simulate_echo(codes,t_vecs,dop_Hz=0.0,range_m=dr,bw=bw,sr=sr)
    z_diff_r=(z0-z_dr)/dr    

    z_ddop=simulate_echo(codes,t_vecs,dop_Hz=ddop,range_m=0.0,bw=bw,sr=sr)
    z_diff_dop=(z0-z_ddop)/ddop
    
    t_l=len(z_dr)
    A=np.zeros([t_l,2],dtype=np.complex64)
    A[:,0]=z_diff_r
    A[:,1]=z_diff_dop
    S=np.real(np.linalg.inv(np.dot(np.transpose(np.conj(A)),A))/snr)

    return np.sqrt(np.diag(S))


def precalculate_dr(txlen, bw, ipp=20e-3, n_ipp=20, n_interp=20):
    enrs = 10.0**np.linspace(-3,10,num=n_interp)
    drs = np.zeros(n_interp)
    ddops = np.zeros(n_interp)    
    for ei, s in tqdm(enumerate(enrs)):
        dr, ddop = lin_error(enr=s,txlen=txlen,bw=bw,ipp=ipp,n_ipp=n_ipp)
        drs[ei] = dr
        ddops[ei] = ddop
    return enrs, drs, ddops


class LinearizedCoded(Errors):
    VARIABLES = [
        'range', 
        'doppler',
        'range_rate',
    ]

    def __init__(self, tx, seed=None, cache_folder=None, diameter=None):
        super().__init__(seed=seed)

        self.diameter = diameter

        bw = tx.bandwidth
        txlen = tx.pulse_length*1e6 # pulse length in microseconds
        ipp = tx.ipp # pulse length in microseconds
        n_ipp = int(tx.n_ipp) # pulse length in microseconds
        self.freq = tx.beam.frequency

        fname = f'MCerr_n{n_ipp}_tx{int(txlen)}_ipp{int(ipp*1e6)}_bw{int(bw)}.h5'

        if cache_folder is None:
            try:
                with importlib.resources.path('sorts.data', fname) as pth:
                    h = h5py.File(pth,'r')
                    enrs = np.copy(h['enrs'][()])
                    drs = np.copy(h['drs'][()])
                    ddops = np.copy(h['ddops'][()])
                    h.close()
            except:
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

        self.rfun = scipy.interpolate.interp1d(np.log10(enrs),np.log10(drs))
        self.dopfun = scipy.interpolate.interp1d(np.log10(enrs),np.log10(ddops))

    def range(self, data, snr):

        dr = 10.0**(self.rfun(np.log10(snr)))

        # if object diameter is larger than range error, make it at least as big as target
        if self.diameter is not None:
            if len(dr.shape) == 0:
                if dr < self.diameter:
                    dr = self.diameter
            else:
                dr[dr < self.diameter] = self.diameter

        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*dr


    def _get_ddop(self, snr):
        ddop = 10.0**(self.dopfun(np.log10(snr)))

        # Unknown doppler shift due to ionosphere can be up to 0.1 Hz,
        # estimate based on typical GNU Ionospheric tomography receiver phase curves.
        if len(ddop.shape) == 0:
            if ddop < 0.1:
                ddop = 0.1
        else:
            ddop[ddop < 0.1] = 0.1
        return ddop


    def doppler(self, data, snr):
        '''Range rate in m/s
        '''
        ddop = self._get_ddop(snr)
        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*ddop


    def range_rate(self, data, snr):
        '''Range rate in m/s
        '''
        ddop = self._get_ddop(snr)
        rr_std = scipy.constants.c*ddop/self.freq/2.0
        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*rr_std





class LinearizedCodedIonospheric(LinearizedCoded):


    def __init__(self, tx, seed=None, cache_folder=None, diameter=None):
        super().__init__(tx, seed=seed, cache_folder=cache_folder, diameter=diameter)

        r = np.array([0.0,1000.0,1000000.0])
        e = np.array([0.0,0.1,100.0])*150.0
        self.iono_errfun = scipy.interpolate.interp1d(r,e)


    def range(self, data, snr, tx_range):
        '''Two way range in meters
        '''

        dr = 10.0**(self.rfun(np.log10(snr)))

        # add ionospheric error
        dr = np.sqrt(dr**2.0 + self.iono_errfun(tx_range/1e3)**2.0)

        if self.diameter is not None:
            if len(dr.shape) == 0:
                if dr < self.diameter:
                    dr = self.diameter
            else:
                dr[dr < self.diameter] = self.diameter

        self.set_numpy_seed()
        return data + np.random.randn(*data.shape)*dr

