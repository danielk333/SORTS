#include "measurements.h"
#include "signals.h"

/*void init_snr_computations(
	void	(*_get_ranges) 				(double **_ranges_tx, double **_ranges_rx),
	void	(*_get_antenna_properties) 	(double **_tx_gain, double **_rx_gain, double **_tx_power),
	void	(*_get_radar_properties) 	(double **_tx_wavelength, double **_rx_wavelength, double **_ipp, double **_pulse_len, double **_duty_cycle, double **_coh_integration_bandwidth, double **_noise, double **_time_slice),
	void 	(*_get_observability)		(int *_is_observable, int _txi, int _rxi),
	void 	(*_get_stop_condition)		(int *_stop, int _txi, int _rxi),
	void 	(*_get_snr_pointers)		(double ***_snr, double ***_snr_incoherent, int ***_keep_data),
	void 	(*_get_rcs_pointer)			(double *_rcs))
{
	python_measurements_callback_functions.get_ranges 				= _get_ranges;

	python_measurements_callback_functions.get_antenna_properties 	= _get_antenna_properties;
	python_measurements_callback_functions.get_radar_properties 	= _get_radar_properties;

	python_measurements_callback_functions.get_observability 		= _get_observability;
	python_measurements_callback_functions.get_stop_condition 		= _get_stop_condition;
	python_measurements_callback_functions.get_result_pointers 		= _get_result_pointers;
};

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
	double *_noise,
	double *_snr,
	double *_snr_inch,
	int* _keep_data,
	double _object_diameter,
	double _radar_albedo,
	double _object_spin_period,
	double _min_snr_db,
	int _doppler_spread_integrated_snr,
	int _snr_limit,
	int _n_time_points)
{
	int t_index;
	t_index = -1;

	// main computation loop
	for(int ti = 0; ti < _n_time_points; ti++)
	{
		double delay;
		double ipp_f;
		double snr_modulation;

		_snr[ti] = 0;

		if(_keep_data[ti] == 1)
		{
			snr_modulation = 1.0;
			if((t_index < _n_time_points - 1 && _t_dirs[ti] >= _t[t_index+1]) || ti == -1)
			{
				t_index++;

		        if(ipp_f <= _pulse_length[t_index])
		        {
		            snr_modulation = ipp_f/_pulse_length[t_index];
		        }
		        else if(ipp_f >= _ipp[t_index] - _pulse_length[t_index])
		        {
		            snr_modulation = (_ipp[t_index] - ipp_f)/_pulse_length[t_index];
		        }
			}

			delay = (_ranges_tx[ti] + _ranges_rx[ti])/C_VACUUM; // compute range delay with signal between tx/rx
	        ipp_f = fmod(delay, _ipp[t_index]);

	        printf("iteration %d\n", ti);
	        printf("delay %f\n", delay);
	        printf("_ipp[ti] %f\n", _ipp[t_index]);
	        printf("_ranges_rx[ti] %f\n", _ranges_rx[ti]);
	        printf("_ranges_tx[ti] %f\n", _ranges_tx[ti]);

	        // check if target is in radars blind range
	        // assume synchronized transmitting assume decoding of partial pulses is possible and linearly decreases signal strength

	        
	        printf("snr_modulation %f\n", snr_modulation);

	        if(_doppler_spread_integrated_snr == 1)
	        {
				doppler_spread_hard_target_snr(
	                    _t_slice[t_index], 
	                    _object_spin_period, 
	                    _tx_gain[ti], 
	                    _rx_gain[ti],
	                    _tx_wavelength[t_index],
	                    _power[t_index],
	                    _ranges_tx[ti], 
	                    _ranges_rx[ti],
	                    _duty_cycle[t_index],
	                    _object_diameter, 
	                    _coh_int_bw[t_index],
	                    _noise[t_index],
	                    _radar_albedo,
	                    &_snr[ti],
	                    &_snr_inch[ti]);
	        }
	        else
	        {
	        	hard_target_snr(
	                _tx_gain[ti],
	                _rx_gain[ti],
	                _tx_wavelength[t_index],
	                _power[t_index],
	                _ranges_tx[ti],
	                _ranges_rx[ti],
	                _object_diameter,
	                _coh_int_bw[t_index],
	                _noise[t_index],
	                _radar_albedo,
	                &_snr[ti]);

	        	_snr_inch[ti] = -1.0;
			}

			printf("_tx_gain[ti] %f\n", _tx_gain[ti]);
			printf("_rx_gain[ti] %f\n", _rx_gain[ti]);
			printf("_snr[ti] %f\n", _snr[ti]);
	        printf("_snr_inch[ti] %f\n", _snr_inch[ti]);

			if(_snr_limit == 1)
			{
				double snr_db;

				if (_snr[ti] < 1e-9)
				{
	                snr_db = 0.0/0.0;
				}
	            else
	            {
	                snr_db = 10.0*log10(_snr[ti]);
	            }

	            if (isnan(snr_db) || isinf(snr_db) || snr_db < _min_snr_db)
	            {
	                _keep_data[ti] = 1;
	                _snr[ti] = 0;
	            }
	            else
	            {
	                _keep_data[ti] = 1;
	            }

	            printf("_keep_data[ti] %d\n", _keep_data[ti]);
	            printf("_min_snr_db %f\n", _min_snr_db);
	       		printf("snr_db %f\n\n", snr_db);
			}
		}
	}
}

void compute_gain(
	double *_t,
	double *_t_dirs,
	double *_gain_tx,
	double *_gain_rx,
	int *_msk,
	int _n_time_points,
	int _n_tx,
	int _n_rx,
	double(*_callback_compute_gain_tx)(int, int, int),
	double(*_callback_compute_gain_rx)(int, int, int, int)
	)
{
	int t_index;
	int msk_rx_counter;

	t_index = 0;
	
	for(int ti = 0; ti < _n_time_points; ti++)
	{
		if(t_index < _n_time_points - 1 && _t_dirs[ti] >= _t[t_index+1])
			t_index++;

		for(int txi = 0; txi < _n_tx; txi++)
		{
			msk_rx_counter = 0;

			for(int rxi = 0; rxi < _n_rx; rxi++)
			{
				if(_msk[rxi*_n_tx + txi] == 1)
				{
					msk_rx_counter++;
					_gain_rx[(rxi*_n_tx + txi)*_n_time_points + ti] = _callback_compute_gain_rx(rxi, txi, ti, t_index);
				}
			}

			if(msk_rx_counter > 0) // if at least one rx station sees an object
				_gain_tx[txi*_n_time_points + ti] = _callback_compute_gain_tx(txi, ti, t_index);
		}
	}
}