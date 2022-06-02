#include "controllers.h"

int normalize_direction_controls(double *orientation_controls, int(*_get_sub_control_array_size_callback)(int, int, int, int))
{
	/**
    	Compute the normalized beam direction vectors.
    	
    	The input format of the beam direction vectors are :
    	    - Tx/Rx stations :
    	        - dim 0: station id
    	        - dim 1: associated station id (if it exists, else the index will be 0)
   		        - dim 2: array of time slices
    	        - dim 3: points per time slice
    	        - dim 4: x, y, z coordinate axis
   	        
    	TODO -> implementation in C/C++
    	TODO -> implement callback python functions to support non-homogeneous arrays of controls
    **/
	
    int n_stations = _get_sub_control_array_size_callback(-1, -1, -1, -1);
    int current_index = 0;

    for(int station_id = 0; station_id < n_stations; station_id++)
    {
    	int n_associated_stations = _get_sub_control_array_size_callback(station_id, -1, -1, -1);
        
        for(int associated_station_id = 0; associated_station_id < n_associated_stations; associated_station_id++)
        {
        	int n_time_points = _get_sub_control_array_size_callback(station_id, associated_station_id, -1, -1);
            
            for(int ti = 0; ti < n_time_points; ti++)
            {
            	int n_points_per_slice = _get_sub_control_array_size_callback(station_id, associated_station_id, ti, -1);
                
                for(int t_slice_id = 0; t_slice_id < n_points_per_slice; t_slice_id++)
                {
                	int vector_size = _get_sub_control_array_size_callback(station_id, associated_station_id, ti, t_slice_id);
                	double norm = compute_norm(orientation_controls, 3, current_index);
                    printf("norm = %f\n", norm);
                	for(int i = 0; i < vector_size; i++)
                    {
                        printf("%d : ormalizing %f -> ", current_index, orientation_controls[current_index]);

                		orientation_controls[current_index] = orientation_controls[current_index]/norm;
                        printf("%f\n", orientation_controls[current_index]);

                        current_index += 1;
                    }

                    printf("normalized vector %d : [%f, %f, %f]\n", current_index, orientation_controls[current_index-3], orientation_controls[current_index-2], orientation_controls[current_index-1]);
                }
            }
        }
    }
       
    return 1;
}

double compute_norm(double* _vector, int _size, int _start_index)
{
	double norm = 0;

	for(int i = 0; i < _size; i++)
		norm += pow(_vector[_start_index+i], 2);

	return sqrt(norm);
}
