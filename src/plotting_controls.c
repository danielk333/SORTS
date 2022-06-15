#include "plotting_controls.h"

void init_plotting_controls(
	void (*_get_pointing_direction)(int, int, int, double*, int),
	int (*_get_n_dirs_per_time_slice)(int, int, int, int),
	void (*_save_pointing_direction_arrays)(double*, double*, int, int, int))
{
	manager_python_functions_callback.get_pointing_direction = _get_pointing_direction;
	manager_python_functions_callback.get_n_dirs_per_time_slice = _get_n_dirs_per_time_slice;
	manager_python_functions_callback.save_pointing_direction_arrays = _save_pointing_direction_arrays;
}

void flatten_directions(int _n_tx, int _n_rx, int _n_time_points)
{
	double** pointing_directions_tx;
	double** pointing_directions_rx;
	double* pointing_directions_tmp;

	int n_points_tx;
	int n_points_rx;
	int n_dirs_per_tslice;

	pointing_directions_tx = (double**)malloc(3*sizeof(double*));
	pointing_directions_rx = (double**)malloc(3*sizeof(double*));

	for(int i = 0; i < 3; i++)
	{
		pointing_directions_tx[i] = (double*)malloc(1*sizeof(double));
		pointing_directions_rx[i] = (double*)malloc(1*sizeof(double));
	}

	pointing_directions_tmp = (double*)malloc(1*sizeof(double));

	n_points_tx = 0;
	n_points_rx = 0;

	// flatten tx directions array
	// flatten tx directions array
	for(int txi = 0; txi < _n_tx; txi++)
	{
		int *n_repeats_tx = (int*)malloc(_n_time_points*sizeof(int));

		for(int rxi = 0; rxi < _n_rx; rxi++)
		{
			for(int ti = 0; ti < _n_time_points; ti++)
			{
				n_dirs_per_tslice = manager_python_functions_callback.get_n_dirs_per_time_slice(txi, rxi, ti, 0);
				n_repeats_tx[ti] = n_dirs_per_tslice;

				pointing_directions_tmp = (double*)realloc(pointing_directions_tmp, n_dirs_per_tslice*3*sizeof(double));
				manager_python_functions_callback.get_pointing_direction(txi, rxi, ti, pointing_directions_tmp, 0);

				for(int i = 0; i < 3; i++)
				{
					pointing_directions_rx[i] = (double*)realloc(pointing_directions_rx[i], (n_points_rx + n_dirs_per_tslice)*sizeof(double));
					
					for(int j = 0; j < n_dirs_per_tslice; j++)
					{ 
 						pointing_directions_rx[i][n_points_rx + j] = pointing_directions_tmp[i + j*3];
						//printf("pointing_directions_rx[%d][%d] = %f\n", i, n_points_rx+j, pointing_directions_rx[i][n_points_rx+j]);
					}
				}

				n_points_rx += n_dirs_per_tslice;
			}
		}

		for(int ti = 0; ti < _n_time_points; ti++)
		{
			n_dirs_per_tslice = manager_python_functions_callback.get_n_dirs_per_time_slice(txi, 0, ti, 1);

			pointing_directions_tmp = (double*)realloc(pointing_directions_tmp, n_dirs_per_tslice*3*sizeof(double));
			manager_python_functions_callback.get_pointing_direction(txi, 0, ti, pointing_directions_tmp, 1);

			for(int i = 0; i < 3; i++)
			{
				pointing_directions_tx[i] = (double*)realloc(pointing_directions_tx[i], (n_points_tx + n_repeats_tx[ti])*sizeof(double));
				//printf("n rep : %d\n", n_repeats_tx[ti]);

				for(int j = 0; j < n_repeats_tx[ti]; j++)
				{

					pointing_directions_tx[i][n_points_tx + j] = pointing_directions_tmp[i + (int)(j*n_dirs_per_tslice/n_repeats_tx[ti]*3)];
					//printf("pointing_directions_tx[%d][%d] = %f\n", i, n_points_tx+j, pointing_directions_tx[i][n_points_tx+j]);
				}
			}

			n_points_tx += n_repeats_tx[ti];
		}

		free(n_repeats_tx);
	}

	for(int i = 0; i < 3; i++)
		manager_python_functions_callback.save_pointing_direction_arrays(pointing_directions_tx[i], pointing_directions_rx[i], n_points_tx, n_points_rx, i);

	for(int i = 0; i < 3; i++)
	{
		free(pointing_directions_tx[i]);
		free(pointing_directions_rx[i]);
	}

	free(pointing_directions_tx);
	free(pointing_directions_rx);		
	free(pointing_directions_tmp);
}




