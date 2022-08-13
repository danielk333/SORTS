from sorts import signals
from sorts.radar import signals_c_wrapper as signals_old
import numpy as np

from sorts import profiling

p = profiling.Profiler()

print("doppler_spread_hard_target_snr performances")

N = 10000

t_obs = 10.0
spin_period = 2.0
gain_tx = np.random.rand(N)*100
gain_rx = np.random.rand(N)*100

f = 960e6
c = 3e8

wavelength = np.random.rand(N)*2*c/f

power_tx = (np.random.rand(N))*100e6

range_tx_m = (np.random.rand(N))*2000e3
range_rx_m = (np.random.rand(N))*2000e3

duty_cycle=0.27
diameter=15.7
bandwidth=11.3
rx_noise_temp=189
radar_albedo=0.154

signal_power = 265e3*(np.random.rand(N))
noise_power = 500*(np.random.rand(N))
epsilon=0.05
incoherent_integration_time=3600.0

snr = 100*(np.random.rand(N))

results_doppler_spread_hard_target_snr = 0
results_incoherent_snr = 0
results_hard_target_diameter = 0
results_hard_target_snr = 0

for i in range(N):
    # test doppler_spread_hard_target_snr
    p.start("doppler_spread_hard_target_snr:python")
    snr1, inc_snr1 = signals_old.doppler_spread_hard_target_snr(
        t_obs, 
        spin_period, 
        gain_tx[i], 
        gain_rx[i],
        wavelength[i],
        power_tx[i],
        range_tx_m[i], 
        range_rx_m[i])

    p.stop("doppler_spread_hard_target_snr:python")

    p.start("doppler_spread_hard_target_snr:c")
    snr2, inc_snr2 = signals.doppler_spread_hard_target_snr(
        t_obs, 
        spin_period, 
        gain_tx[i], 
        gain_rx[i],
        wavelength[i],
        power_tx[i],
        range_tx_m[i], 
        range_rx_m[i])
    p.stop("doppler_spread_hard_target_snr:c")

    results_doppler_spread_hard_target_snr += np.sum(snr1-snr2) + np.sum(inc_snr1-inc_snr2)
    
    # test doppler_spread_hard_target_snr
    p.start("incoherent_snr:python")
    snr1, inc_snr1, tobs1 = signals_old.incoherent_snr(
        signal_power[i], 
        noise_power[i], 
        epsilon, 
        bandwidth, 
        incoherent_integration_time)
    p.stop("incoherent_snr:python")

    p.start("incoherent_snr:c")
    snr2, inc_snr2, tobs2 = signals.incoherent_snr(
        signal_power[i], 
        noise_power[i], 
        epsilon, 
        bandwidth, 
        incoherent_integration_time)

    p.stop("incoherent_snr:c")

    results_incoherent_snr += np.sum(snr1-snr2) + np.sum(inc_snr1-inc_snr2) + np.sum(tobs1-tobs2)

    # test hard_target_diameter
    p.start("hard_target_diameter:python")
    d1 = signals_old.hard_target_diameter(
    			gain_tx[i], 
                gain_rx[i],
                wavelength[i],
                power_tx[i],
                range_tx_m[i], 
                range_rx_m[i],
                snr[i], 
                bandwidth,
                rx_noise_temp,
                radar_albedo)
    p.stop("hard_target_diameter:python")

    p.start("hard_target_diameter:c")
    d2 = signals.hard_target_diameter(
    			gain_tx[i], 
                gain_rx[i],
                wavelength[i],
                power_tx[i],
                range_tx_m[i], 
                range_rx_m[i],
                snr[i], 
                bandwidth,
                rx_noise_temp,
                radar_albedo)

    p.stop("hard_target_diameter:c")

    results_hard_target_diameter += np.sum(d1-d2)
    print("p", d1)
    print("c", d2)

    # test hard_target_snr_clib
    p.start("hard_target_snr:python")
    d1 = signals_old.hard_target_snr(
                gain_tx[i], 
                gain_rx[i],
                wavelength[i],
                power_tx[i],
                range_tx_m[i], 
                range_rx_m[i],
                diameter, 
                bandwidth,
                rx_noise_temp,
                radar_albedo)
    p.stop("hard_target_snr:python")

    p.start("hard_target_snr:c")
    d2 = signals.hard_target_snr(
                gain_tx[i], 
                gain_rx[i],
                wavelength[i],
                power_tx[i],
                range_tx_m[i], 
                range_rx_m[i],
                diameter, 
                bandwidth,
                rx_noise_temp,
                radar_albedo)
    p.stop("hard_target_snr:c")

    results_hard_target_snr += np.sum(d1-d2)

    print("Test iteration ", i, "/", N)

print("doppler_spread_hard_target_snr : ", results_doppler_spread_hard_target_snr)
print("incoherent_snr : ", results_incoherent_snr)
print("hard_target_diameter : ", results_hard_target_diameter)
print("hard_target_snr : ", results_hard_target_snr)

print(p)