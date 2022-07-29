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
	int _n_dirs,
	int _ctrl_start_id,
	int _pdirs_start_id
	)
{
	int t_index;
	int t_measurements;

	t_index = _ctrl_start_id - 1;

	// main computation loop
	for(int ti = _pdirs_start_id; ti < _n_dirs + _pdirs_start_id; ti++)
	{
		double delay;
		double ipp_f;
		double snr_modulation;

		t_measurements = ti - _pdirs_start_id;
		_snr[t_measurements] = 0;

		snr_modulation = 1.0;
		if((t_index < _n_time_points + _ctrl_start_id - 1 && _t_dirs[ti] >= _t[t_index+1]) || ti == -1)
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

		delay = (_ranges_tx[t_measurements] + _ranges_rx[t_measurements])/C_VACUUM; // compute range delay with signal between tx/rx
        ipp_f = fmod(delay, _ipp[t_index]);

        // printf("\niteration %d\n", ti);

        // printf("_t %f\n", _t[t_index]);
        // printf("_t_dirs %f\n", _t_dirs[ti]);
        // printf("_t_slice %f\n", _t_slice[t_index]);
        // printf("_pulse_length %f\n", _pulse_length[t_index]);
        // printf("_ipp %f\n", _ipp[t_index]);
        // printf("_tx_gain %f\n", _tx_gain[t_measurements]);
        // printf("_rx_gain %f\n", _rx_gain[t_measurements]);
        // printf("_tx_wavelength %f\n", _tx_wavelength[t_index]);
        // printf("_power %f\n", _power[t_index]);
        // printf("_ranges_tx %f\n", _ranges_tx[t_measurements]);
        // printf("_ranges_rx %f\n", _ranges_rx[t_measurements]);
        // printf("_duty_cycle %f\n", _duty_cycle[t_index]);
        // printf("_coh_int_bw %f\n", _coh_int_bw[t_index]);

        // check if target is in radars blind range
        // assume synchronized transmitting assume decoding of partial pulses is possible and linearly decreases signal strength

        
        // printf("snr_modulation %f\n", snr_modulation);

        if(_doppler_spread_integrated_snr == 1)
        {
			doppler_spread_hard_target_snr(
                    _t_slice[t_index], 
                    _object_spin_period, 
                    _tx_gain[t_measurements], 
                    _rx_gain[t_measurements],
                    _tx_wavelength[t_index],
                    _power[t_index],
                    _ranges_tx[t_measurements], 
                    _ranges_rx[t_measurements],
                    _duty_cycle[t_index],
                    _object_diameter, 
                    _coh_int_bw[t_index],
                    _noise,
                    _radar_albedo,
                    &_snr[t_measurements],
                    &_snr_inch[t_measurements]);
        }
        else
        {
        	hard_target_snr(
                _tx_gain[t_measurements],
                _rx_gain[t_measurements],
                _tx_wavelength[t_index],
                _power[t_index],
                _ranges_tx[t_measurements],
                _ranges_rx[t_measurements],
                _object_diameter,
                _coh_int_bw[t_index],
                _noise,
                _radar_albedo,
                &_snr[t_measurements]);

        	_snr_inch[t_measurements] = -1.0;
		}

		// printf("_snr[ti] %f\n", _snr[t_measurements]);
  //       printf("_snr_inch[ti] %f\n", _snr_inch[t_measurements]);

		if(_snr_limit == 1)
		{
			double snr_db;

			if (_snr[t_measurements] < 1e-9)
			{
                snr_db = 0.0/0.0;
			}
            else
            {
                snr_db = 10.0*log10(_snr[t_measurements]);
            }

            if (isnan(snr_db) || isinf(snr_db) || snr_db < _min_snr_db)
            {
                _detection[t_measurements] = 0;
                _snr[t_measurements] = 0;
            }
            else
            {
                _detection[t_measurements] = 1;
            }
		}
	}
}

void compute_gain(
	double *_t,
	double *_t_dirs,
	int _n_points,
	int _n_dirs,
	int _ctrl_start_id,
	int _pdirs_start_id,
	double(*_callback_compute_gain_tx)(int, int),
	double(*_callback_compute_gain_rx)(int, int)
	)
{
	int t_index;
	int flag_fov;

	t_index = _ctrl_start_id-1;
	for(int ti = _pdirs_start_id; ti < _n_dirs + _pdirs_start_id; ti++) /// [txi, rxi, ti]
	{
		if(t_index < _ctrl_start_id + _n_points - 1 && _t_dirs[ti] >= _t[t_index+1])
			t_index++;

		_callback_compute_gain_tx(ti, t_index);
		_callback_compute_gain_rx(ti, t_index);
	}
}

void get_max_snr_measurements(
	double *_t,
	double *_t_meas,
	double *_snr,
	int* _inds,
	int _n_time_points,
	int _n_inds)
{
	int index;

	index = -1;
	_inds[0] = 0;

	// loop through all the snr array
	for(int ti = 0; ti < _n_time_points; ti++)
	{
		// increment the index of the final measurement array if we have left the time slice
		if(index < _n_inds - 1 && _t_meas[ti] >= _t[index+1])
		{
			index++;
			_inds[index] = ti;
		}

		//printf("ti = %f (%d) -> snr = %f / (current max %f (t_slice %f - %d)) - index %d\n", _t_meas[ti], ti, _snr[ti], _snr[_inds[index]], _t[index], index, _inds[index]);
		if(_snr[ti] > _snr[_inds[index]])
		{
			_inds[index] = ti;
		}
	};
}
