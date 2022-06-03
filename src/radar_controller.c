#include "radar_controller.h"

int check_time_slice_overlap(double* t_controller, double* time_slice, int n, int* indices, void(*_callback_save_array)(int*, int))
{
    int n_indices = 0;

    for (int index = 0; index < n-1; index++)
    {
        if (t_controller[index+1] - t_controller[index] + 1e-10 < time_slice[index])
        {   
            
            indices = (int*)realloc(indices, (n_indices+1) * sizeof(int));
            indices[n_indices] = index;
            n_indices++;
        }
    }

    _callback_save_array(indices, n_indices);

    return 1;
}