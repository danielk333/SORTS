#include "radar_controller.h"

int check_time_slice_overlap(double* _t_controller, double* _time_slice, int _n, int* _indices, void(*_callback_save_array)(int*, int))
{
    int n_indices = 0;

    for (int index = 0; index < _n-1; index++)
    {
        if (_t_controller[index+1] - _t_controller[index] + 1e-10 < _time_slice[index])
        {   
            
            _indices = (int*)realloc(_indices, (n_indices+1) * sizeof(int));
            _indices[n_indices] = index;
            n_indices++;
        }
    }

    _callback_save_array(_indices, n_indices);

    return 1;
}

int is_active_time_slice(
    double _t, 
    double* _t_controller, 
    double* _time_slice, 
    int _n)
{
    for (int index = 0; index < _n; index++)
    {
        double dt = _t_controller[index] - _t;

        if (dt + 1e-10 < _time_slice[index] && dt >= 0.0)
        {   
            return 1;
        }
    }

    return 0;
}