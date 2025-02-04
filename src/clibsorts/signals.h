// inclusion guard
#ifndef SIGNALS_H_
#define SIGNALS_H_

#include <stdio.h>
#include <stdlib.h>

#include <math.h>

/** Boltzman constant : \f$ k_B=1.380649 \cdot 10^{-23} J \f$ **/
#define BOLTZMAN_CONSTANT 1.380649e-23


/** Speed of light in vacuum : \f$ c=299 792 458 m/s \f$ **/
#define C_VACUUM 299792458 // meters per seconds

/*!
 * \section desc Description
 * This function computes the coherent and incoherent signal-to-noise (SNR) ratios for a spinning rigid
 * target by taking into account the doppler shift.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 * 		The current implementation of the function does not support array inputs over
 * 		all parameters
 * \endverbatim
 *
 * @param _t_obs: 				`(double)`  measurement duration (s)
 * @param _spin_period: 		`(double)` 	rotation period of the object being observed (s)
 * @param _gain_tx: 			`(double*)` transmit antenna gain, linear (-)
 * @param _gain_rx: 			`(double*)` receiver antenna gain, linear (-)
 * @param _wavelength: 			`(double)`  radar wavelength (m)
 * @param _power_tx: 			`(double)`  transmit power (W)
 * @param _range_tx_m: 			`(double*)` range from transmitter to target (m)
 * @param _range_rx_m: 			`(double*)` range from target to receiver (m)
 * @param _duty_cycle: 			`(double)` 	radar measurement duty cycle (-)
 * @param _diameter: 			`(double)` 	object diameter (m)
 * @param _bandwidth: 			`(double)` 	effective receiver noise bandwidth (Hz)
 * @param _rx_noise_temp: 		`(double)` 	eceiver noise temperature (K)
 * @param _radar_albedo: 		`(double)` 	radar albedo of the object beaing observed (-)
 * @param _snr_coh: 			`(double*)` coherent signal-to-noise ration computation results (-)
 * @param _snr_incoh: 			`(double*)` incoherent signal-to-noise ration computation results (-)
 * @param _N: 					`(int)` 	incoherent signal-to-noise ration computation results (-)
 *
 * \section ressources External resources
 * - *D. Kastinen et al., Radar observability of near-Earth objects with EISCAT 3D, 2020,*
 *
 *
 *
 *
  */
void doppler_spread_hard_target_snr_vectorized(
	double _t_obs,
	double _spin_period,
	double *_gain_tx,
	double *_gain_rx,
	double _wavelength,
	double _power_tx,
	double *_range_tx_m,
	double *_range_rx_m,
	double _duty_cycle,
	double _diameter,
	double _bandwidth,
	double _rx_noise_temp,
	double _radar_albedo,
	double *_snr_coh,
	double *_snr_incoh,
	int _N);


/**
 * \section desc Description
 * This function computes the coherent and incoherent signal-to-noise (SNR) ratios for a spinning rigid
 * target by taking into account the doppler shift.
 *
 * @param _t_obs: 			`(double)`  measurement duration  (s)
 * @param _spin_period: 	`(double)`  rotation period of the object being observed (s)
 * @param _gain_tx: 		`(double)`  transmit antenna gain, linear  (-)
 * @param _gain_rx: 		`(double)`  receiver antenna gain, linear  (-)
 * @param _wavelength 		`(double)`  radar wavelength (m)
 * @param _power_tx: 		`(double)`  transmit power (W)
 * @param _range_tx_m: 		`(double)`  range from transmitter to target (m)
 * @param _range_rx_m: 		`(double)`  range from target to receiver (m)
 * @param _duty_cycle: 		`(double)`  radar measurement duty cycle  (-)
 * @param _diameter: 		`(double)`  object diameter (m)
 * @param _bandwidth: 		`(double)`  effective receiver noise bandwidth (Hz)
 * @param _rx_noise_temp: 	`(double)`  receiver noise temperature (K)
 * @param _radar_albedo: 	`(double)`  radar albedo of the object beaing observed  (-)
 * @param _snr_coh: 		`(double*)` coherent signal-to-noise ration computation results (-)
 * @param _snr_incoh: 		`(double*)` incoherent signal-to-noise ration computation results (-)
 *
 * @returns: `(void)`
 *
 * \subsection ressources External resources
 * 	 - D. Kastinen et al., Radar observability of near-Earth objects with EISCAT 3D, 2020,
 *
 *
 *
 */
void doppler_spread_hard_target_snr(
	double _t_obs,
	double _spin_period,
	double _gain_tx,
	double _gain_rx,
	double _wavelength,
	double _power_tx,
	double _range_tx_m,
	double _range_rx_m,
	double _duty_cycle,
	double _diameter,
	double _bandwidth,
	double _rx_noise_temp,
	double _radar_albedo,
	double *_snr_coh,
	double *_snr_incoh);


/**
 * \section desc Description
 * Computes the incoherent signal to noise ratio and the minimal observation time required
 *
 * @param _signal_power: 				`(double)`  signal power (W)
 * @param _noise_power: 				`(double)`  noise power (W)
 * @param _epsilon: 					`(double)`  statistical significance criterion used for a detection (-)
 * @param _bandwidth: 					`(double)`  measurement bandwidth (Hz)
 * @param _incoherent_integration_time: `(double)`  incoherent integration time of the measurements (s)
 * @param _snr: 						`(double*)` SNR of coherently integrated measurement (-)
 * @param _snr_incoh: 					`(double*)` SNR of incoherently integrated measurement (-)
 * @param _minimal_observation_time: 	`(double*)` minimal observation time required (s)
 *
 * @returns: `(void)`
 *
 * \subsection ressources External resources
 * 	 - D. Kastinen et al.: Radar observability of near-Earth objects with EISCAT 3D, 2020
 *
 *
 *
 */
void incoherent_snr(
	double _signal_power,
	double _noise_power,
	double _epsilon,
	double _bandwidth,
	double _incoherent_integration_time,
	double *_snr,
	double *_snr_incoh,
	double *_minimal_observation_time);


/**
 * \section desc Description
 * This function estimates the diamter of a hard target based on the signal-to-noise ratio (energy-to-noise).
 * Assume a smooth transition between Rayleigh and optical scattering. Ignore Mie regime and use either optical
 * or Rayleigh scatter.
 *
 * @param _gain_tx: 		`(double)`  transmit antenna gain, linear (-)
 * @param _gain_rx: 		`(double)`  receiver antenna gain, linear (-)
 * @param _wavelength: 		`(double)`  radar wavelength (m)
 * @param _power_tx: 		`(double)`  transmit power (W)
 * @param _range_tx_m: 		`(double)`  range from transmitter to target (m)
 * @param _range_rx_m: 		`(double)`  range from target to receiver (m)
 * @param _snr: 			`(double)`  object signal to noise ratio (-)
 * @param _bandwidth: 		`(double)`  effective receiver noise bandwidth (Hz)
 * @param _rx_noise_temp: 	`(double)`  receiver noise temperature (K)
 * @param _radar_albedo: 	`(double)`  radar albedo (-)
 * @param _diameter: 		`(double*)` diameter (m)
 * @param _N: 				`(int)` 	array sizes
 *
 * @returns `(void)`
 *
 * \subsection ressources External resources
 *  - Markkanen et.al., 1999
 *
 *
 *
 */
void hard_target_diameter_vectorized(
	double *_gain_tx,
	double *_gain_rx,
	double _wavelength,
	double _power_tx,
	double *_range_tx_m,
	double *_range_rx_m,
	double *_snr,
	double _bandwidth,
	double _rx_noise_temp,
	double _radar_albedo,
	double *_diameter,
	int _N);


/**
 * \section desc Description
 * This function estimates the diamter of a hard target based on the signal-to-noise ratio (energy-to-noise).
 * Assume a smooth transition between Rayleigh and optical scattering. Ignore Mie regime and use either optical
 * or Rayleigh scatter.
 *
 * @param _gain_tx: 		`(double)`  transmit antenna gain, linear (-)
 * @param _gain_rx: 		`(double)`  receiver antenna gain, linear (-)
 * @param _wavelength: 		`(double)`  radar wavelength (m)
 * @param _power_tx: 		`(double)`  transmit power (W)
 * @param _range_tx_m: 		`(double)`  range from transmitter to target (m)
 * @param _range_rx_m: 		`(double)`  range from target to receiver (m)
 * @param _snr: 			`(double)`  object signal to noise ratio (-)
 * @param _bandwidth: 		`(double)`  effective receiver noise bandwidth (Hz)
 * @param _rx_noise_temp: 	`(double)`  receiver noise temperature (K)
 * @param _radar_albedo: 	`(double)`  radar albedo (-)
 * @param _diameter: 		`(double)`  target diameter (m)
 *
 * @returns `void`
 *
 * \subsection ressources External resources
 * - Markkanen et.al., 1999
 *
 *
 */
void hard_target_diameter(
	double _gain_tx,
	double _gain_rx,
	double _wavelength,
	double _power_tx,
	double _range_tx_m,
	double _range_rx_m,
	double _snr,
	double _bandwidth,
	double _rx_noise_temp,
	double _radar_albedo,
	double *_diameter);


/**
 * \section desc Description
 * This function determines the *Signal-to-Noise Ratio* SNR (or energy-to-noise ratio) of a **hard target**.
 * It assumes a smooth transition between the Rayleigh and optical scattering regimes. The Mie regime
 * is ignored.
 *
 * @param _gain_tx:			`(double)`  transmit antenna gain (-).
 * @param _gain_rx: 		`(double)`  receiver antenna gain (-).
 * @param _wavelength: 		`(double)`  radar wavelength (m).
 * @param _power_tx: 		`(double)`  transmit power (W).
 * @param _range_tx_m: 		`(double)`  range from transmitter to target (m).
 * @param _range_rx_m: 		`(double)`  range from target to receiver (m).
 * @param _diameter: 		`(double)`  object diameter (m).
 * @param _bandwidth: 		`(double)`  effective receiver noise bandwidth (Hz).
 * @param _rx_noise_temp:	`(double)`  receiver noise temperature (K).
 * @param _radar_albedo: 	`(double)`  radar albedo (-).
 * @param _snr: 			`(double*)` target SNR computation results (-).
 * @param _N: 				`(int)` 	size of the arrays.
 *
 * @returns `void`
 *
 * \subsection ressources External resources
 *  - Markkanen et.al., 1999
 *
 *
 */
void hard_target_snr_vectorized(
	double *_gain_tx,
	double *_gain_rx,
	double _wavelength,
	double _power_tx,
	double *_range_tx_m,
	double *_range_rx_m,
	double _diameter,
	double _bandwidth,
	double _rx_noise_temp,
	double _radar_albedo,
	double *_snr,
	int _N);

/**
 * \section desc Description
 * This function determines the *signal-to-noise ratio* (energy-to-noise) for a **hard target**.
 * It assumes a smooth transition between the Rayleigh and optical scattering regimes. The Mie regime
 * is ignored.
 *
 * @param _gain_tx:			`(double)`  transmit antenna gain.
 * @param _gain_rx: 		`(double)`  receiver antenna gain.
 * @param _wavelength: 		`(double)`  radar wavelength (meters).
 * @param _power_tx: 		`(double)`  transmit power (W).
 * @param _range_tx_m: 		`(double)`  range from transmitter to target (meters).
 * @param _range_rx_m: 		`(double)`  range from target to receiver (meters).
 * @param _diameter: 		`(double)`  object diameter (meters).
 * @param _bandwidth: 		`(double)`  effective receiver noise bandwidth.
 * @param _rx_noise_temp: 	`(double)`  receiver noise temperature (K).
 * @param _radar_albedo: 	`(double)`  radar albedo.
 * @param _snr: 			`(double*)` Signal-to-noise ratio computation results.
 *
 * @returns `void`
 *
 * \subsection ressources External resources
 *  - Markkanen et.al., 1999
 *
 *
 */
void hard_target_snr(
	double _gain_tx,
	double _gain_rx,
	double _wavelength,
	double _power_tx,
	double _range_tx_m,
	double _range_rx_m,
	double _diameter,
	double _bandwidth,
	double _rx_noise_temp,
	double _radar_albedo,
	double *_snr
);

#endif //SIGNALS_H_
