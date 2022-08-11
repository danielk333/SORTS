#include "measurements.h"
#include "signals.h"

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
	int _n_dirs
	)
{
	int t_index;

	t_index = 0;

	// main computation loop
	for(int ti = 0; ti < _n_dirs; ti++)
	{
		double delay;
		double ipp_f;
		double snr_modulation;

		_snr[ti] = 0;

		snr_modulation = 1.0;
		if((t_index < _n_time_points - 1 && _t_dirs[ti] >= _t[t_index+1]) || ti == 0)
		{
			t_index = get_next_control_time_index(_t, _t_dirs[ti], t_index, _n_time_points);

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

        // printf("\niteration %d\n", ti);

        // printf("_t %f\n", _t[t_index]);
        // printf("_t_dirs %f\n", _t_dirs[ti]);
        // printf("_t_slice %f\n", _t_slice[t_index]);
        // printf("_pulse_length %f\n", _pulse_length[t_index]);
        // printf("_ipp %f\n", _ipp[t_index]);
        // printf("_tx_gain %f\n", _tx_gain[ti]);
        // printf("_rx_gain %f\n", _rx_gain[ti]);
        // printf("_tx_wavelength %f\n", _tx_wavelength[ti]);
        // printf("_power %f\n", _power[t_index]);
        // printf("_ranges_tx %f\n", _ranges_tx[ti]);
        // printf("_ranges_rx %f\n", _ranges_rx[ti]);
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
                    _tx_gain[ti], 
                    _rx_gain[ti],
                    _tx_wavelength[ti],
                    _power[t_index],
                    _ranges_tx[ti], 
                    _ranges_rx[ti],
                    _duty_cycle[t_index],
                    _object_diameter, 
                    _coh_int_bw[t_index],
                    _noise,
                    _radar_albedo,
                    &_snr[ti],
                    &_snr_inch[ti]);
        }
        else
        {
        	hard_target_snr(
                _tx_gain[ti],
                _rx_gain[ti],
                _tx_wavelength[ti],
                _power[t_index],
                _ranges_tx[ti],
                _ranges_rx[ti],
                _object_diameter,
                _coh_int_bw[t_index],
                _noise,
                _radar_albedo,
                &_snr[ti]);

        	_snr_inch[ti] = -1.0;
		}

		// printf("_snr[ti] %f\n", _snr[ti]);
  //       printf("_snr_inch[ti] %f\n", _snr_inch[ti]);

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
                _detection[ti] = 0;
                _snr[ti] = 0;
            }
            else
            {
                _detection[ti] = 1;
            }
		}
	}
}

void compute_gain(
	double *_t,
	double *_t_dirs,
	int _n_points,
	int _n_dirs,
	double(*_callback_compute_gain_tx)(int, int),
	double(*_callback_compute_gain_rx)(int, int)
	)
{
	int t_index;
	t_index = 0;
	for(int ti = 0; ti < _n_dirs; ti++) /// [txi, rxi, ti]
	{
		if(t_index < _n_points - 1 && _t_dirs[ti] >= _t[t_index+1])
			t_index = get_next_control_time_index(_t, _t_dirs[ti], t_index, _n_points);

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
		if(index < _n_inds && _t_meas[ti] >= _t[index+1])
		{
			index = get_next_control_time_index(_t, _t_meas[ti], index, _n_inds);
			_inds[index] = ti;
		}

		//printf("ti = %f (%d) -> snr = %f / (current max %f (t_slice %f - %d)) - index %d\n", _t_meas[ti], ti, _snr[ti], _snr[_inds[index]], _t[index], index, _inds[index]);
		if(_snr[ti] > _snr[_inds[index]])
		{
			_inds[index] = ti;
		}
	};
}

int get_next_control_time_index(
	double* _t,
	double _t_dir,
	int _ti,
	int _N
	)
{
	while(_ti < _N-1 && _t[_ti + 1] <= _t_dir)
	{
		_ti++;
	}
	//printf("_t[ti]=%f, tref=%f, ti=%d, _n=%d\n", _t[_ti], _t_dir, _ti, _N);

	return _ti;
}