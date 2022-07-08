#include <iostream>
#include <cmath>
#include <complex>

#define PI 3.141592

extern "C" 
{
	void pulse_function_no_code(double _t, double _A, double _f, double _T, double *_pulse_real, double *_pulse_imaginary);
	void pulse_function_code(double _t, double _A, double _f, double _T, double *_pulse_real, double *_pulse_imaginary);

	double barker13(double _t, double _amplitude, double _duration);

	void autocorrelation_function_code(double *_autocorrelation_function, double* _t, double *_measured_signal_r, double *_measured_signal_i, double _f0, double _fmix,double _fmin, double _fmax, double _tsmin, double _tsmax, double _pulse_duration, double _x_norm, int _N, int _N_pulses, int _N_autocorrelation, void(*_it_counter_callback)(int, int));
	void autocorrelation_function_no_code(double *_autocorrelation_function, double* _t, double *_measured_signal_r, double *_measured_signal_i, double _f0, double _fmix,double _fmin, double _fmax, double _tsmin,double _tsmax, double _pulse_duration, double _x_norm,  int _N,  int _N_pulses,  int _N_autocorrelation, void(*_it_counter_callback)(int, int));
	void autocorrelation_function_no_code(double *_autocorrelation_function, double* _t, double *_measured_signal_r, double *_measured_signal_i, double _f0, double _fmix,double _fmin, double _fmax, double _tsmin,double _tsmax, double _pulse_duration, double _x_norm,  int _N,  int _N_pulses,  int _N_autocorrelation, void(*_it_counter_callback)(int, int));

	void correlate(double *_measured_signal_r, double *_measured_signal_i, double *_ref_signal_r, double *_ref_signal_i, double *_correlated_signal, double _x_norm, int _N, int _N_IPP);
}

void pulse_function_no_code(double _t, double _A, double _f, double _T, double *_pulse_real, double *_pulse_imaginary)
{
	if(_t > _T || _t < 0)
	{
		*_pulse_imaginary = 0;
		*_pulse_real = 0;		
	}
    else
    {
    	double w = 2*PI*_f;
        
        std::complex<double> i(0, 1);
        std::complex<double> signal = _A*std::exp(i*w*_t);

        *_pulse_imaginary = signal.imag();
		*_pulse_real = signal.real();	
    }
}

void pulse_function_code(double _t, double _A, double _f, double _T, double *_pulse_real, double *_pulse_imaginary)
{
	if(_t > _T || _t < 0)
	{
		*_pulse_imaginary = 0;
		*_pulse_real = 0;		
	}
    else
    {
    	double w = 2*PI*_f;
        
        std::complex<double> i(0, 1);

        std::complex<double> signal = barker13(_t, 1, _T) * _A * std::exp(i*w*_t);

        *_pulse_imaginary = signal.imag();
		*_pulse_real = signal.real();	
    }
}

double barker13(double _t, double _amplitude, double _duration)
{
	if(_t >= _duration || _t < 0) return 0;
	else
	{
		double code[13] = {1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
		
		int j = (int)(_t/_duration*13);
		return code[j] * _amplitude;
	}
}

void autocorrelation_function_code(
	double *_autocorrelation_function,
	double* _t, 
	double *_measured_signal_r, 
	double *_measured_signal_i, 
	double _f0,
	double _fmix,
	double _fmin,
	double _fmax,
	double _tsmin,
	double _tsmax,
	double _pulse_duration, 
	double _x_norm, 
	int _N, 
	int _N_pulses, 
	int _N_autocorrelation,
	void(*_it_counter_callback)(int, int))
{
	for(int i = 0; i < _N_autocorrelation; i++)
	{
		double freq = (_fmax-_fmin)*(((double)i)/((double)_N_autocorrelation)) + _fmin;

		//std::cout << "i = " << i << std::endl;

		for(int j = 0; j < _N_autocorrelation; j++)
		{
			//std::cout << "j = " << j << std::endl;
			double t_shift = (_tsmax-_tsmin)*(((double)j)/((double)_N_autocorrelation)) + _tsmin;
			std::complex<double> sum = 0;

			for(int k = 0; k < _N_pulses; k++)
			{
				for(int n = 0; n < _N; n++)
				{
					double tester_signal_r, tester_signal_i;

					//std::cout << "n' = " << (n-j)%_N << std::endl;

					pulse_function_code(_t[n]-t_shift, 1, _f0+freq-_fmix, _pulse_duration, &tester_signal_r, &tester_signal_i);

					sum += std::complex<double>(_measured_signal_r[n + k*_N], -_measured_signal_i[n + k*_N])*std::complex<double>(tester_signal_r, tester_signal_i);
				}
			}

			_autocorrelation_function[i*_N_autocorrelation+j] = std::abs(sum.real())/_x_norm;
		}
		_it_counter_callback(i, _N_autocorrelation);
	}
}

void autocorrelation_function_no_code(
	double *_autocorrelation_function,
	double* _t, 
	double *_measured_signal_r, 
	double *_measured_signal_i, 
	double _f0,
	double _fmix,
	double _fmin,
	double _fmax,
	double _tsmin,
	double _tsmax,
	double _pulse_duration, 
	double _x_norm, 
	int _N, 
	int _N_pulses, 
	int _N_autocorrelation,
	void(*_it_counter_callback)(int, int))
{
	for(int i = 0; i < _N_autocorrelation; i++)
	{
		//std::cout << "i = " << i << std::endl;
		double freq = (_fmax-_fmin)*(((double)i)/((double)_N_autocorrelation)) + _fmin;

		for(int j = 0; j < _N_autocorrelation; j++)
		{
			double t_shift = (_tsmax-_tsmin)*(((double)j)/((double)_N_autocorrelation)) + _tsmin;
			std::complex<double> sum = 0;
			//std::cout << "j = " << j << std::endl;

			for(int k = 0; k < _N_pulses; k++)
			{
				for(int n = 0; n < _N; n++)
				{
					double tester_signal_r, tester_signal_i;
					pulse_function_no_code(_t[n]-t_shift, 1, _f0 + freq - _fmix, _pulse_duration, &tester_signal_r, &tester_signal_i);
					//std::cout << "n' = " << n << "/N = " << _N << " -> t = " << _t[(n-j)%_N] << std::endl;

					sum += std::complex<double>(_measured_signal_r[n + k*_N], -_measured_signal_i[n + k*_N])*std::complex<double>(tester_signal_r, tester_signal_i);
				}
			}

			_autocorrelation_function[i*_N_autocorrelation+j] = std::abs(sum.real())/_x_norm;
		}
		_it_counter_callback(i, _N_autocorrelation);
	}
}

void correlate(
	double *_measured_signal_r, 
	double *_measured_signal_i, 
	double *_ref_signal_r, 
	double *_ref_signal_i, 
	double *_correlated_signal, 
	double _x_norm, 
	int _N, 
	int _N_IPP)
{
	for(int i = 0; i < _N; i++)
	{
		std::complex<double> sum = 0;

		for(int ipp = 0; ipp <  _N_IPP; ipp++)
		{
			for (int j = i; j < _N; j++)
			{
				std::complex<double> tmp = std::complex<double>(_measured_signal_r[j + ipp*_N_IPP], -_measured_signal_i[j + ipp*_N_IPP]) * std::complex<double>(_ref_signal_r[j - i + ipp*_N_IPP], _ref_signal_i[j - i]);
				if(j > i && tmp.real() == 0 && tmp.imag() == 0) break;

				sum += tmp;
			}
		}

		_correlated_signal[i] = std::sqrt(std::pow(sum.real(), 2) + std::pow(sum.real(), 2))/_x_norm;
	}
}