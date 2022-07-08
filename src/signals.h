#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#define BOLTZMAN_CONSTANT 1.380649e-23
#define C_VACUUM 299792458 // meters per seconds

void doppler_spread_hard_target_snr_vectorized(double _t_obs, double _spin_period, double *_gain_tx, double *_gain_rx,double _wavelength,double _power_tx, double *_range_tx_m, double *_range_rx_m,double _duty_cycle,double _diameter, double _bandwidth,double _rx_noise_temp,double _radar_albedo, double *_snr_coh, double *_snr_incoh, int _N);
void doppler_spread_hard_target_snr(double _t_obs, double _spin_period, double _gain_tx, double _gain_rx,double _wavelength,double _power_tx, double _range_tx_m, double _range_rx_m,double _duty_cycle,double _diameter, double _bandwidth, double _rx_noise_temp,double _radar_albedo, double *_snr_coh, double *_snr_incoh);

void incoherent_snr(double _signal_power, double _noise_power, double _epsilon, double _bandwidth, double _incoherent_integration_time, double *_snr, double *_snr_incoh, double *_minimal_observation_time);

void hard_target_diameter_vectorized(double *_gain_tx, double *_gain_rx, double _wavelength, double _power_tx, double *_range_tx_m,  double *_range_rx_m, double *_snr, double _bandwidth, double _rx_noise_temp, double _radar_albedo, double *_diameter, int _N);
void hard_target_diameter(double _gain_tx, double _gain_rx, double _wavelength, double _power_tx, double _range_tx_m,  double _range_rx_m, double _snr, double _bandwidth, double _rx_noise_temp, double _radar_albedo, double *_diameter);

void hard_target_snr_vectorized(double *_gain_tx, double *_gain_rx, double _wavelength, double _power_tx, double *_range_tx_m, double *_range_rx_m, double _diameter, double _bandwidth, double _rx_noise_temp, double _radar_albedo, double *_snr, int _N);
void hard_target_snr(double _gain_tx, double _gain_rx, double _wavelength, double _power_tx, double _range_tx_m, double _range_rx_m, double _diameter, double _bandwidth, double _rx_noise_temp, double _radar_albedo, double *_snr);
