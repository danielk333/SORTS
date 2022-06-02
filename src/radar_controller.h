#include <stdio.h>
#include <stdlib.h>

#include <math.h>

// function definition
int check_time_slice_overlap(double* t_controller, double* time_slice, int n, int* indices, void(*_callback_save_array)(int*, int));
