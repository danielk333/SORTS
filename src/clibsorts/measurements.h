#include <stdio.h>
#include <stdlib.h>
/*
typedef struct python_measurements_callback_functions_holder_struct
{
	void	(*get_ranges) 				(double **_ranges_tx, double **_ranges_rx, int _n_tx, int _n_rx, int _time_array_size);

	void	(*get_antenna_properties) 	(double **_tx_gain, double **_rx_gain, double **_tx_power, int _n_tx, int _n_rx, int _time_array_size);
	void	(*get_radar_properties) 	(double **_tx_wavelength, double **_rx_wavelength, double **_ipp, double **_pulse_len, double **_duty_cycle, double **_coh_integration_bandwidth, double **_noise, double **_time_slice, int _n_tx, int _n_rx, int _time_array_size);

	void 	(*get_observability)		(int *_is_observable, int _txi, int _rxi, int _time_array_size),
	void 	(*get_stop_condition)		(int *_stop, int _txi, int _rxi, int _time_array_size),

	void 	(*get_snr_pointers)		(double *_rcs);
	void 	(*get_rcs_pointers)		(double ***_snr, double ***_snr_incoherent, int ***_keep_data);
} python_measurements_callback_functions_holder;

python_measurements_callback_functions_holder python_measurements_callback_functions;

*/

void compute_measurement_snr(
	double *_t,
	double *_t_dirs,
	double *_t_slice,
	double *_pulse_length,
	double *_ipp,
	double *_tx_gain,
	double *_rx_gain,
	double *_tx_wavelength,
	double *_power,
	double *_ranges_tx,
	double *_ranges_rx,
	double *_duty_cycle,
	double *_coh_int_bw,
	double *_snr,
	double *_snr_inch,
	int* _detection,
	double _noise,
	double _object_diameter,
	double _radar_albedo,
	double _object_spin_period,
	double _min_snr_db,
	int _doppler_spread_integrated_snr,
	int _snr_limit,
	int _n_time_points,
	int _n_dirs);

void compute_gain(
	double *_t,
	double *_t_dirs,
	int _n_points,
	int _n_dirs,
	double(*_callback_compute_gain_tx)(int, int),
	double(*_callback_compute_gain_rx)(int, int)
	);

void get_max_snr_measurements(
	double *_t,
	double *_t_meas,
	double *_snr,
	int* _inds,
	int _n_time_points,
	int _n_inds);

int get_next_control_time_index(
	double* _t,
	double _t_dir,
	int _ti,
	int _N
	);