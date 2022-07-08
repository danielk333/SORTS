# -*- coding: utf-8 -*-
"""
Created on Sun May  8 09:48:19 2022

@author: Thomas Maynadié
"""

import ctypes
import numpy as np

from . import clibcoh

def encode_barker13(t, amplitude, duration):
    clibcoh.barker13.argtypes = [ctypes.c_double, 
                             ctypes.c_double, 
                             ctypes.c_double]
    
    clibcoh.barker13.restype = ctypes.c_double
    
    return clibcoh.barker13(ctypes.c_double(t), ctypes.c_double(amplitude), ctypes.c_double(duration))


def autocorrelation_function(measured_signal, t, f0, fmix, doppler_freqs, tshifts, pulse_duration, x_norm, N, N_pulses, N_AF_points, code=True):
    AF = np.ndarray([N_AF_points, N_AF_points], dtype=np.float64)
    
    COUNTERITFNC = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)

    def it_counter(it, itmax):
        print("Iteration {0}/{1}".format(it, itmax))    
    
    it_counter_c = COUNTERITFNC(it_counter)
    
    if code == True:                
        clibcoh.autocorrelation_function_code.argtypes = [np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=AF.ndim, shape=AF.shape),
                                                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape) ,
                                                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=measured_signal.ndim, shape=measured_signal.shape) ,
                                                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=measured_signal.ndim, shape=measured_signal.shape) ,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                COUNTERITFNC]
            
        clibcoh.autocorrelation_function_code(AF, 
                                     t.astype(np.float64),
                                     measured_signal.real.astype(np.float64), 
                                     measured_signal.imag.astype(np.float64), 
                                     ctypes.c_double(f0), 
                                     ctypes.c_double(fmix), 
                                     ctypes.c_double(min(doppler_freqs)), 
                                     ctypes.c_double(max(doppler_freqs)),  
                                     ctypes.c_double(min(tshifts)), 
                                     ctypes.c_double(max(tshifts)), 
                                     ctypes.c_double(pulse_duration),
                                     ctypes.c_double(x_norm),
                                     ctypes.c_int(N),
                                     ctypes.c_int(N_pulses),
                                     ctypes.c_int(N_AF_points),
                                     it_counter_c)
        
    else:        
        clibcoh.autocorrelation_function_no_code.argtypes = [np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=AF.ndim, shape=AF.shape),
                                                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape) ,
                                                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=measured_signal.ndim, shape=measured_signal.shape) ,
                                                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=measured_signal.ndim, shape=measured_signal.shape) ,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_double,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                COUNTERITFNC]
            
        clibcoh.autocorrelation_function_no_code(AF, 
                                     t.astype(np.float64),
                                     measured_signal.real.astype(np.float64), 
                                     measured_signal.imag.astype(np.float64), 
                                     ctypes.c_double(f0), 
                                     ctypes.c_double(fmix),                                      
                                     ctypes.c_double(min(doppler_freqs)), 
                                     ctypes.c_double(max(doppler_freqs)), 
                                     ctypes.c_double(min(tshifts)), 
                                     ctypes.c_double(max(tshifts)), 
                                     ctypes.c_double(pulse_duration),
                                     ctypes.c_double(x_norm),
                                     ctypes.c_int(N),
                                     ctypes.c_int(N_pulses),
                                     ctypes.c_int(N_AF_points),
                                     it_counter_c)
        
    return AF
        
def create_radar_pulse(amplitude, pulse_function, radar_frequency, fmix, N, inter_pulse_period, pulse_duration, code=False):
    t = np.linspace(0, inter_pulse_period, N)
    signal = pulse_function(t, amplitude, radar_frequency, fmix, pulse_duration, code=code)
    
    return t, signal

@np.vectorize
def pulse_function(t, amplitude, radar_frequency, fmix, pulse_duration, code=False):
    im = ctypes.c_double(0)
    re = ctypes.c_double(0)
    
    double_p = ctypes.POINTER(ctypes.c_double)
    
    if code == False:
        clibcoh.pulse_function_no_code.argtypes = [ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               double_p,
                                               double_p]
            
        clibcoh.pulse_function_no_code(ctypes.c_double(t),
                                ctypes.c_double(amplitude),
                                ctypes.c_double(radar_frequency-fmix),
                                ctypes.c_double(pulse_duration),
                                re,
                                im)

    else:
        clibcoh.pulse_function_code.argtypes = [ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               double_p,
                                               double_p]
        
        clibcoh.pulse_function_code(ctypes.c_double(t),
                                ctypes.c_double(amplitude),
                                ctypes.c_double(radar_frequency-fmix),
                                ctypes.c_double(pulse_duration),
                                ctypes.byref(re),
                                ctypes.byref(im))
        
    return complex(re.value, im.value)
    
@np.vectorize
def create_echo(t, amplitude, f0, fmix, df, time_shift, pulse_duration, code=False):
    im = ctypes.c_double(0)
    re = ctypes.c_double(0)
    
    double_p = ctypes.POINTER(ctypes.c_double)
    
    new_duration = pulse_duration*f0/(df+f0)
    
    if code == False:
        clibcoh.pulse_function_no_code.argtypes = [ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               double_p,
                                               double_p]
            
        clibcoh.pulse_function_no_code(ctypes.c_double(t-time_shift),
                                ctypes.c_double(amplitude),
                                ctypes.c_double(f0+df-fmix),
                                ctypes.c_double(new_duration),
                                re,
                                im)

    else:
        clibcoh.pulse_function_code.argtypes = [ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               ctypes.c_double,
                                               double_p,
                                               double_p]
            
        clibcoh.pulse_function_code(ctypes.c_double(t-time_shift),
                                ctypes.c_double(amplitude),
                                ctypes.c_double(f0+df-fmix),
                                ctypes.c_double(new_duration),
                                ctypes.byref(re),
                                ctypes.byref(im))
        
    return complex(re.value, im.value)


def correlate(measured_signal, reference_signal, N_IPP, x_norm):
    if np.size(reference_signal) != int(np.size(measured_signal)/N_IPP):
        raise(ValueError(f"measured_signal (size {np.size(measured_signal)}) and reference_signal (size {np.size(reference_signal)}) must be of the sale size."))

    N = len(reference_signal)
    correlated_signal = np.zeros(N, dtype=np.float64)
    
    clibcoh.correlate.argtypes = [
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=measured_signal.ndim, shape=measured_signal.shape),
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=measured_signal.ndim, shape=measured_signal.shape) ,
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=reference_signal.ndim, shape=reference_signal.shape) ,
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=reference_signal.ndim, shape=reference_signal.shape) ,
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=correlated_signal.ndim, shape=correlated_signal.shape) ,
                                            ctypes.c_double, 
                                            ctypes.c_int, 
                                            ctypes.c_int,
                                            ]
        
    clibcoh.correlate(
                                measured_signal.real.astype(np.float64), 
                                measured_signal.imag.astype(np.float64), 
                                reference_signal.real.astype(np.float64),
                                reference_signal.imag.astype(np.float64),
                                correlated_signal,
                                ctypes.c_double(x_norm),
                                ctypes.c_int(N),
                                ctypes.c_int(N_IPP))
        
    return correlated_signal

