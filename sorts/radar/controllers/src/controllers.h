#include <stdio.h>
#include <stdlib.h>

#include <math.h>

// function definition
int normalize_direction_controls(double *orientation_controls, int(*_get_sub_control_array_size_callback)(int, int, int, int));
double compute_norm(double* _vector, int _size, int _start_index);
