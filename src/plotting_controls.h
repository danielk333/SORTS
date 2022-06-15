#include <stdio.h>
#include <stdlib.h>

struct manager_python_functions_struct {
	void 	(*get_pointing_direction)(int, int, int, double*, int);
	int 	(*get_n_dirs_per_time_slice)(int, int, int, int);
	void 	(*save_pointing_direction_arrays)(double*, double*, int, int, int);
};

typedef struct manager_python_functions_struct manager_python_functions;

manager_python_functions manager_python_functions_callback;

void init_plotting_controls(
	void 	(*_get_pointing_direction)(int, int, int, double*, int),
	int 	(*_get_n_dirs_per_time_slice)(int, int, int, int),
	void 	(*_save_pointing_direction_arrays)(double*, double*, int, int, int)
	);

void flatten_directions(int _n_rx, int _n_tx, int _n_time_points);
