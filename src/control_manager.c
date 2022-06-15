#include "control_manager.h"

void init_manager(
    int(*_get_control_period_id)(int, int),
    int(*_get_control_array_size)(int, int),
	int(*_get_control_parameters)(int , int , int, double*, double*, int*),
	void(*_get_control_arrays)(int, int, double* , double* , int*),
	void(*_save_new_control_arrays)(int* , int* , int)
    )
{
	callback_manager_python_functions.get_control_period_id = _get_control_period_id;
	callback_manager_python_functions.get_control_array_size = _get_control_array_size;
	callback_manager_python_functions.get_control_arrays = _get_control_arrays;
	callback_manager_python_functions.save_new_control_arrays = _save_new_control_arrays;
	
	callback_manager_python_functions.get_control_parameters = _get_control_parameters;
}


double run_manager(
	int _period_id,
	int _n_controls_in_period, 
	double _t_start, 
	double _t_end)
{
	int* control_period_ids;
	int* new_control_id;
	int* new_control_time_id;
	int* closest_control_points_indices;

	int new_control_array_size;
	int end_flag;
	double t;

	control_point next_control_point;

	// allocate memory for new control arrays
	end_flag = 0;
	new_control_array_size = 0;
	new_control_id = (int*)malloc(1 * sizeof(int));
	new_control_time_id = (int*)malloc(1 * sizeof(int));

	closest_control_points_indices = (int*)malloc(_n_controls_in_period * sizeof(int));

	// get indices of controls subarrays inside the current manager period
	control_period_ids = (int*)malloc(_n_controls_in_period * sizeof(int));
	get_control_period_indices(_period_id, control_period_ids, _n_controls_in_period);
	
	// intialize time cursor position
	t = _t_start;

	// loop through the whole current time array
	while(t < _t_end && end_flag == 0)
	{
		int found_new_point;
		next_control_point.control_id = -1;

		// gather all the closest next time points
		get_next_time_points(t, _n_controls_in_period, &next_control_point, closest_control_points_indices, control_period_ids, &found_new_point);

		// we have found the next control point
		if (found_new_point == 1 && check_conflicts(&t, _n_controls_in_period, next_control_point, closest_control_points_indices, control_period_ids) == 1 
		&& next_control_point.control_id != -1) // if a new compatible control time point has been identified
		{
			if(next_control_point.t + next_control_point.t_slice > _t_end) // if the time slice is on the boundary
			{
				// if the time slice is on the boundary, checks if there is a conflict with the next manager period
				found_new_point = boundary(&t, _period_id, _n_controls_in_period, next_control_point);
				end_flag = 1; // we  have reached the end of the manager period
			}
			
			if(found_new_point == 1)
			{
				new_control_id = (int*)realloc(new_control_id, (new_control_array_size+1)*sizeof(int));
				new_control_time_id = (int*)realloc(new_control_time_id, (new_control_array_size+1)*sizeof(int));

				t = next_control_point.t + next_control_point.t_slice;

				new_control_id[new_control_array_size] = next_control_point.control_id;
				new_control_time_id[new_control_array_size] = closest_control_points_indices[next_control_point.control_id];
				new_control_array_size += 1;
			}
		}
		else
		{
			if(t == next_control_point.t + next_control_point.t_slice)
				end_flag = 1;
		}	
	}

	free(closest_control_points_indices);
	free(control_period_ids);

	callback_manager_python_functions.save_new_control_arrays(new_control_time_id, new_control_id, new_control_array_size);

	free(new_control_time_id);
	free(new_control_id);

	return t;
}

void get_next_time_points(
		double _t,
		int _n_controls_in_period,
		control_point *_next_control_point,
		int *_closest_control_points_indices,
		int *_control_period_ids,
		int *_found_new_point
	)
{
	int point_counter;
	control_point tmp_control_point;

	point_counter = 0;

	// get next closest time points
	for(int ctrl_id = 0; ctrl_id < _n_controls_in_period; ctrl_id++)
	{
		int time_array_size;

		// get control time array size
		time_array_size = callback_manager_python_functions.get_control_array_size(ctrl_id, _control_period_ids[ctrl_id]);

		if(time_array_size > 0)
		{
			// get controls subarrays
			double* time_array = (double*)malloc(time_array_size * sizeof(double));
			double* t_slice = (double*)malloc(time_array_size * sizeof(double));
			int* priority = (int*)malloc(time_array_size * sizeof(int));

			callback_manager_python_functions.get_control_arrays(ctrl_id, _control_period_ids[ctrl_id], time_array, t_slice, priority);

			// store next closest time point id for each time array
			_closest_control_points_indices[ctrl_id] = get_next_control_point_id(_t, time_array, time_array_size);

			if(point_counter == 0)
			{
				*_found_new_point = (_closest_control_points_indices[0] == -1 ? 0 : 1);

				tmp_control_point.t = time_array[_closest_control_points_indices[0]];
				tmp_control_point.t_slice = t_slice[_closest_control_points_indices[0]];
				tmp_control_point.priority = priority[_closest_control_points_indices[0]];
				tmp_control_point.control_id = 0;
			}
			else
			{
				// store the closest time point
				if(_closest_control_points_indices[ctrl_id] != -1 && (time_array[_closest_control_points_indices[ctrl_id]] <= tmp_control_point.t || time_array[_closest_control_points_indices[ctrl_id]] == tmp_control_point.t && priority[_closest_control_points_indices[ctrl_id]] < tmp_control_point.priority || *_found_new_point != 1)) // if first point of array or if there is a closest time point
				{
					*_found_new_point = 1;
					
					tmp_control_point.t = time_array[_closest_control_points_indices[ctrl_id]];
					tmp_control_point.t_slice = t_slice[_closest_control_points_indices[ctrl_id]];
					tmp_control_point.priority = priority[_closest_control_points_indices[ctrl_id]];
					tmp_control_point.control_id = ctrl_id;
				}
			}

			free(time_array);
			free(t_slice);
			free(priority);

			point_counter++;
		}
	}

	if(*_found_new_point == 1)
	{
		_next_control_point->t = tmp_control_point.t;
		_next_control_point->t_slice = tmp_control_point.t_slice;
		_next_control_point->priority = tmp_control_point.priority;
		_next_control_point->control_id = tmp_control_point.control_id;
	}
}

int check_conflicts(double *_t, int _n_controls_in_period, control_point _next_control_point, int* _closest_control_points_indices, int *_control_period_ids)
{
	int iflag;

	int next_priority;
	int point_exists;

	double next_control_time_point;
	double next_t_slice;

	iflag = 1;

	// check for conflicts with every other controls with higher priority
	for(int ctrl_id = 0; ctrl_id < _n_controls_in_period; ctrl_id++)
	{
		if (ctrl_id != _next_control_point.control_id && _closest_control_points_indices[ctrl_id] != -1)
		{
			// get control time array
			point_exists = callback_manager_python_functions.get_control_parameters(ctrl_id, _control_period_ids[ctrl_id], _closest_control_points_indices[ctrl_id], &next_control_time_point, &next_t_slice, &next_priority);
			if(point_exists != -1)
			{
				if(next_control_time_point - _next_control_point.t < _next_control_point.t_slice &&  _next_control_point.priority > next_priority) // if we find a new control point with higher priority inside the considered time slice
				{
					iflag = 0;
					
					// get furthest time point which is preventing control from executing 
					if (*_t < next_control_time_point)
					{
						*_t = next_control_time_point;
					}
				}
			}
		}
	}
	return iflag;
}

int get_next_control_point_id(double _t, double* _time_array, int _time_array_size)
{
	int index = -1;

	for(int time_index = 0; time_index < _time_array_size; time_index++)
	{
		if(_time_array[time_index] >= _t)
		{
			index = time_index;
			break;
		}
	}

	return index;
}

void get_control_period_indices(int _manager_period_id, int *_control_period_indices, int _n_controls)
{
	for(int ctrl_id = 0; ctrl_id < _n_controls; ctrl_id++)
		_control_period_indices[ctrl_id] = callback_manager_python_functions.get_control_period_id(ctrl_id, _manager_period_id);
}

int boundary(
	double *_t,
	int _period_id,
	int _n_controls_in_period,
	control_point _next_control_point)
{
	int* control_period_ids;
	int next_priority;
	int found_new_point;

	double next_control_time_point;
	double next_t_slice;

	found_new_point = 1;

	// get indices of controls subarrays inside the current manager period
	control_period_ids = (int*)malloc(_n_controls_in_period * sizeof(int));
	get_control_period_indices(_period_id+1, control_period_ids, _n_controls_in_period);

	for(int ctrl_id = 0; ctrl_id < _n_controls_in_period; ctrl_id++)
	{
		if (ctrl_id != _next_control_point.control_id)
		{
			int point_exists = callback_manager_python_functions.get_control_parameters(ctrl_id, control_period_ids[ctrl_id], 0, &next_control_time_point, &next_t_slice, &next_priority);

			if(point_exists != -1)
			{
				if(next_control_time_point - _next_control_point.t < _next_control_point.t_slice && _next_control_point.priority > next_priority) // if we find a new control point with higher priority inside the considered time slice
				{
					found_new_point = 0;
					
					// get furthest time point which is preventing control from executing 
					if (*_t < next_control_time_point)
					{
						*_t = next_control_time_point;
					}
				}
			}
		}
	}

	return found_new_point;
}