#include "signals.h"


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
        int _N
    )
{
    double power;
    double rx_noise;
    double separator;

    // marks the separation between the Optical/Reighley regimes
    separator = _wavelength/(M_PI*sqrt(3.0));
    rx_noise = BOLTZMAN_CONSTANT * _rx_noise_temp * _bandwidth;

    // compute for each points
    for(int ti = 0; ti < _N; ti++)
    {
        if (_diameter < separator) // Reighley scattering regime
            power = _power_tx*_gain_tx[ti]*_gain_rx[ti]*pow(3.0*M_PI*pow(_diameter, 3.0)/(_wavelength*_range_rx_m[ti]*_range_tx_m[ti]), 2.0)/256.0;
        else // Optical scattering regime
            power = _power_tx*_gain_tx[ti]*_gain_rx[ti]*pow(_wavelength * _diameter / (M_PI*_range_tx_m[ti]*_range_rx_m[ti]), 2.0)/256.0;

        // compute SNR
        _snr[ti] = power*_radar_albedo/rx_noise;
    }
}


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
    )
{
    double power;
    double rx_noise;
    double separator;

    // Rx noise power
    rx_noise = BOLTZMAN_CONSTANT * _rx_noise_temp * _bandwidth;

    // marks the separation between the Optical/Reighley regimes
    separator = _wavelength/(M_PI*sqrt(3.0));

    if (_diameter < separator) // Reighley scattering regime
        power = _power_tx*_gain_tx*_gain_rx*pow(3.0*M_PI/(_wavelength*_range_rx_m*_range_tx_m), 2.0)*pow(_diameter, 6.0) / 256.0;
    else // Optical scattering regime
        power = _power_tx*_gain_tx*_gain_rx*pow(_wavelength*_diameter/(M_PI*_range_tx_m*_range_rx_m), 2.0) / 256.0;

    // compute SNR
    *_snr = power*_radar_albedo/rx_noise;
}


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
            int _N
        )
{
    double diameter_rayleigh;
    double diameter_optical;
    double separatrix;
    double rx_noise;
    double power;

    // Rx noise power
    rx_noise = BOLTZMAN_CONSTANT * _rx_noise_temp * _bandwidth;

    // compute for each time point
    for(int ti = 0; ti < _N; ti++)
    {
        // Rx power of the signal
        power = _snr[ti]*rx_noise/_radar_albedo;

        // compute diameters for each scattering regime
        diameter_rayleigh = pow(256.0* power / (9.0*_power_tx*_gain_tx[ti]*_gain_rx[ti]) * pow(_wavelength *_range_rx_m[ti]*_range_tx_m[ti]/M_PI, 2.0), 1.0/6.0);
        diameter_optical = sqrt(256.0 * pow(M_PI*_range_rx_m[ti]*_range_tx_m[ti]/_wavelength, 2.0) * power/(_power_tx*_gain_tx[ti]*_gain_rx[ti]));

        // compute transition diameter (Rayleigh/Optical)
        separatrix = _wavelength / (M_PI * sqrt(3.0));

        // compute real diameter
        if(diameter_rayleigh < separatrix)
            _diameter[ti] = diameter_rayleigh;
        else
            _diameter[ti] = diameter_optical;
    }
}


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
            double *_diameter
        )
{
    double diameter_rayleigh;
    double diameter_optical;
    double separatrix;
    double rx_noise;
    double power;

    // noise power
    rx_noise = BOLTZMAN_CONSTANT * _rx_noise_temp * _bandwidth;

    // received power
    power = _snr*rx_noise/_radar_albedo;

    // compute diameters for each scattering mode
    diameter_rayleigh = pow(256.0*power*pow(_wavelength*_range_rx_m*_range_tx_m/M_PI, 2.0)/(9.0*_power_tx*_gain_tx*_gain_rx), 1.0/6.0);
    diameter_optical = sqrt(256.0*power*pow(M_PI*_range_rx_m*_range_tx_m/_wavelength, 2.0)/(_power_tx * _gain_tx * _gain_rx));

    // compute transition diameter (Rayleigh/Optical)
    separatrix = _wavelength/(M_PI * sqrt(3.0));

    // compute real diameter
    if(diameter_rayleigh < separatrix && diameter_optical < separatrix)
        *_diameter = diameter_rayleigh;
    if(diameter_rayleigh >= separatrix && diameter_optical >= separatrix)
        *_diameter = diameter_optical;
}


void incoherent_snr(
    double _signal_power,
    double _noise_power,
    double _epsilon,
    double _bandwidth,
    double _incoherent_integration_time,
    double *_snr,
    double *_snr_incoh,
    double *_minimal_observation_time
    )
{
    int n_measurement;
    // compute the number of measurements needed
    n_measurement = (int)(_incoherent_integration_time * _bandwidth);

    // results
    *_snr = _signal_power/_noise_power; // Compute the signal to noise ratio
    *_minimal_observation_time = pow((_signal_power + _noise_power)/(_epsilon*_signal_power), 2.0)/_bandwidth; // compute the minimum required observation time needed to reduce the relative error to epsilon as follows
    *_snr_incoh = (*_snr)*sqrt(n_measurement); // Compute the incoherent signal to noise ratio
}


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
        double *_snr_incoh
    )
{
    double doppler_bandwidth;
    double detection_bandwidth;
    double base_int_bandwidth;

    double coh_noise_power;
    double incoh_noise_power;
    double minimal_observation_time;

    double signal_power;
    double rx_noise;
    double h_snr;
    double snr;

    // Compute signal properties
    hard_target_snr(_gain_tx, _gain_rx, _wavelength, _power_tx, _range_tx_m, _range_rx_m, _diameter, _bandwidth, _rx_noise_temp, _radar_albedo, &h_snr);

    // compute the bandwidth of the doppler shifted RADAR echo
    doppler_bandwidth = 4.0*M_PI*_diameter/(_wavelength*_spin_period);
    detection_bandwidth = 0;
    base_int_bandwidth = 0;

    // compute the bandwidth for the coherently and incoherently integrated measurements
    // get detection_bandwidth
    double tmp[3] = {doppler_bandwidth, 1.0/_t_obs, _bandwidth*_duty_cycle};
    for(int k = 0; k < 3; k++)
    {
        if(detection_bandwidth < tmp[k])
        {
            detection_bandwidth = tmp[k];
        }
    }

    // get base_int_bandwidth
    for(int k = 0; k < 2; k++)
    {
        if(base_int_bandwidth < tmp[k])
        {
            base_int_bandwidth = tmp[k];
        }
    }

    rx_noise = BOLTZMAN_CONSTANT * _rx_noise_temp * _bandwidth; // compute the noise measured by the receiver
    signal_power = h_snr * rx_noise; // compute the noised measured by the receiver

    // compute the SNR
    // coherent : effective noise power when using just coherent integration
    coh_noise_power = BOLTZMAN_CONSTANT* _rx_noise_temp * detection_bandwidth/_duty_cycle;
    *_snr_coh = signal_power/coh_noise_power;

    // incoherent : effective noise power when doing incoherent integration and using a good a priori orbital elements
    incoh_noise_power = BOLTZMAN_CONSTANT * _rx_noise_temp * base_int_bandwidth/_duty_cycle;

    incoherent_snr(signal_power, incoh_noise_power, 0.05, base_int_bandwidth, _t_obs, &snr, _snr_incoh, &minimal_observation_time);
}


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
        int _N
    )
{
    double doppler_bandwidth;
    double detection_bandwidth;
    double base_int_bandwidth;

    double coh_noise_power;
    double incoh_noise_power;
    double minimal_observation_time;
    double signal_power;
    double rx_noise;
    double snr;

    double *h_snr;
    h_snr = (double*)malloc(_N*sizeof(double));

    // Compute signal properties
    hard_target_snr_vectorized(_gain_tx, _gain_rx, _wavelength, _power_tx, _range_tx_m, _range_rx_m, _diameter, _bandwidth, _rx_noise_temp, _radar_albedo, h_snr, _N);

    for(int ti = 0; ti < _N; ti++)
    {
        //printf("%f, %f, %f, %f\n", _gain_tx[ti], _gain_rx[ti], _range_tx_m[ti], _range_rx_m[ti]);
        // compute the bandwidth of the doppler shifted RADAR echo
        doppler_bandwidth = 4.0*M_PI*_diameter/(_wavelength*_spin_period);

        detection_bandwidth = 0.0;
        base_int_bandwidth = 0.0;

        // compute the bandwidth for the coherently and incoherently integrated measurements
        // get detection_bandwidth
        double tmp1[3] = {doppler_bandwidth, _bandwidth*_duty_cycle, 1.0/_t_obs};
        for(int k = 0; k < 3; k++)
        {
            if(detection_bandwidth < tmp1[k])
            {
                detection_bandwidth = tmp1[k];
            }
        }

        // get base_int_bandwidth
        double tmp2[2] = {doppler_bandwidth, 1.0/_t_obs};
        for(int k = 0; k < 2; k++)
        {
            if(base_int_bandwidth < tmp2[k])
            {
                base_int_bandwidth = tmp2[k];
            }
        }

        rx_noise = BOLTZMAN_CONSTANT * _rx_noise_temp * _bandwidth; // compute the noise measured by the receiver
        signal_power = h_snr[ti] * rx_noise; // compute the noised measured by the receiver

        // compute the SNR
        // coherent : effective noise power when using just coherent integration
        coh_noise_power = BOLTZMAN_CONSTANT* _rx_noise_temp * detection_bandwidth/_duty_cycle;
        _snr_coh[ti] = signal_power/coh_noise_power;

        // incoherent : effective noise power when doing incoherent integration and using a good a priori orbital elements
        incoh_noise_power = BOLTZMAN_CONSTANT * _rx_noise_temp * base_int_bandwidth/_duty_cycle;

        incoherent_snr(signal_power, incoh_noise_power, 0.05, base_int_bandwidth, _t_obs, &snr, &_snr_incoh[ti], &minimal_observation_time);
    }
}
