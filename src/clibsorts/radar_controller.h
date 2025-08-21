#include <stdio.h>
#include <stdlib.h>

#include <math.h>

// function definition
int check_time_slice_overlap(double* _t_controller, double* _time_slice, int _n, int* _indices, void(*_callback_save_array)(int*, int));
int is_active_time_slice(double _t, double* _t_controller, double* _time_slice, int _n);
