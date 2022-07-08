#include "static_priority_scheduler.h"

void init_static_priority_scheduler(
    int(*_get_control_period_id)(int, int),
	int(*_get_control_parameters)(int , int , int, double*, double*, int*),
	void(*_get_control_arrays)(int, double** , double** , int**, int*),
	void(*_save_new_control_arrays)(int* , int* , int)
    )
{
	/**

	function : init_static_priority_scheduler

	Checks if if there is a point which prevents the execution of the current time slice (i.e if there is an other time slice with higher priority executing during the current time slice)
	in the next schedule perdiod.
	This function is used as a boundary condition ensureing the continuity of the RADAR schedule between two consecutive scheduler periods.

	If there is a time slice with higer priority interrupting the next control point previously found, the algorithm will discard the latter and move the time cursor to the higher 
	priority time slice starting.

	Parameters :
	------------
	_get_control_period_id : int(*)
		callback function pointer used to get the index of the time sub array corresponding to the current time slice

	_get_control_parameters : void(*)
		callback function pointer used to get the control parameters (start time, time slice and priority) at a given index in the given scheduler period

	_get_control_arrays : int(*) 
		callback function pointer used to get the arrays of control parameters  (start time, time slice and priority) at a given scheduler period

	_save_new_control_arrays : void(*)
		callback function pointer used to save the dynamically allocated schedule control sequence 
		
	**/

	callback_static_priority_scheduler_python_functions.get_control_period_id = _get_control_period_id;
	callback_static_priority_scheduler_python_functions.get_control_arrays = _get_control_arrays;
	callback_static_priority_scheduler_python_functions.save_new_control_arrays = _save_new_control_arrays;
	callback_static_priority_scheduler_python_functions.get_control_parameters = _get_control_parameters;
}


double run_static_priority_scheduler(
	int _period_id,
	int _n_controls_in_period, 
	double _t_start, 
	double _t_end)
{
	/**

	function : run_static_priority_scheduler

	This function runs the static priority scheduler algorithm.

	The algorithm uses a time cursor which freely moves through the whole scheduler period. The algorithm recovers the start time of the closest time slice for each control. 
	After getting the closest one, the algorithme then checks if the latter is in conflict with an other time slice with higher priority, if not then the time slice is added 
	to the output control sequence, and if there is a conflict, the algorithm will discard the closest time slice move the time cursor to the starting point of the conflicting 
	time slice of highest priority and will repeat the same procedure. 

	The algorithm ensures the continuity of the schedule between the different scheduler periods by executing a continuity algorithm at the end of each scheduler period 
	(see the function static_priority_scheduler_boundary).

	Parameters :
	------------
	_period_id : int 
		index of the current scheduler period

	_n_controls_in_period : int
		number of distinct controls to be scheduled

	_t_start : double
		time cursor starting point. Usually, _t_start corresponds to the starting point of the current scheduler period.
	
	_t_end : double
		End point at which the algorithm will transition to the next scheduler period.

	Returns :
	---------

	double :
		the function returns the lat position of the scheduler time cursor to ensure continuity with the next scheduler period.
		
	**/
	static_priority_scheduler_control_point next_control_point;

	int 		*control_period_ids;
	int 		*new_control_id;
	int 		*new_control_time_id;
	int 		*closest_control_points_indices;

	int 		new_control_array_size;
	int 		end_flag;
	double 		t;

	// control arrays
	double 		**time_array;
	double 		**time_slices;
	int 		**priorities;
	int 		*control_array_sizes;

	time_array 				= (double**)	malloc(_n_controls_in_period*sizeof(double*));
	time_slices 			= (double**)	malloc(_n_controls_in_period*sizeof(double*));
	priorities 				= (int**)		malloc(_n_controls_in_period*sizeof(int*));
	control_array_sizes 	= (int*)		malloc(_n_controls_in_period*sizeof(int));

	// get control arrays from ythe python library
	callback_static_priority_scheduler_python_functions.get_control_arrays(_period_id, time_array, time_slices, priorities, control_array_sizes);

	// allocate memory for the final control sequence arrays
	// those arrays will be saved back to the python library using a callback function
	end_flag = 0;
	new_control_array_size = 0;
	new_control_id = (int*)malloc(1 * sizeof(int));
	new_control_time_id = (int*)malloc(1 * sizeof(int));

	closest_control_points_indices = (int*)malloc(_n_controls_in_period * sizeof(int));
	
	// intialize the position of the scheduler time cursor
	t = _t_start;

	// loop through the whole current time array
	while(t < _t_end && end_flag == 0)
	{
		int found_new_point;
		next_control_point.control_id = -1;

		// gather all the closest next time points
		static_priority_scheduler_get_next_time_points(t, _n_controls_in_period, &next_control_point, closest_control_points_indices, time_array, time_slices, priorities, control_array_sizes, &found_new_point);

		// If a new time slice is found, the scheduler checks if the latter is in confkic with other time slices
		if (found_new_point == 1 && static_priority_scheduler_check_conflicts(&t, _n_controls_in_period, next_control_point, closest_control_points_indices, time_array, time_slices, priorities, control_array_sizes) == 1 
		&& next_control_point.control_id != -1)
		{
			// if the time slice is on the boundary, a special continuity algorithm is executed
			if(next_control_point.t + next_control_point.t_slice > _t_end) 
			{
				found_new_point = static_priority_scheduler_boundary(&t, _period_id, _n_controls_in_period, next_control_point);
				end_flag = 1; // we  have reached the end of the scheduler period
			}
			
			// if the time slice isn't in conflict with other time slices, the time slice is added to the final control sequence
			if(found_new_point == 1)
			{
				new_control_id = (int*)realloc(new_control_id, (new_control_array_size+1)*sizeof(int));
				new_control_time_id = (int*)realloc(new_control_time_id, (new_control_array_size+1)*sizeof(int));

				t = next_control_point.t + next_control_point.t_slice; // moving the cursor to the end of the time slice

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

	// free the allocated memory
	free(closest_control_points_indices);
	free(control_period_ids);
	free(time_array);
	free(time_slices);
	free(priorities);
	free(control_array_sizes);

	// save the control sequence to python
	callback_static_priority_scheduler_python_functions.save_new_control_arrays(new_control_time_id, new_control_id, new_control_array_size);

	free(new_control_time_id);
	free(new_control_id);

	// return the position of the cursor
	return t;
}

void static_priority_scheduler_get_next_time_points(
		double _t,
		int _n_controls_in_period,
		static_priority_scheduler_control_point *_next_control_point,
		int *_closest_control_points_indices,
		double **_time_array,
		double **_time_slices,
		int **_priorities,
		int *_control_array_size,
		int *_found_new_point
	)
{
	/**

	function : static_priority_scheduler_get_next_time_points

	Gathers the control time slice closest to the current cursor position for each control.

	Parameters :
	------------
	_t : double*
		current position of the time cursor. This cursor represents the current position of the scheduling algorithme in the current scheduler period.
	
	_n_controls_in_period : int
		number of distinct controls to be scheduled
	
	_next_control_point : struct static_priority_scheduler_control_point*
		this structure encapsulates the next control point to be executed. The algorithm checks if there is a time slice with higer priority to be executed during the current time slice _next_control_point
	
	_closest_control_points_indices : int*
		time point index of the closest time slice

	_time_array : double**
		array of control time points

	_time_slices : double**
		array of control time slice durations

	_priorities : double**
		array of control priorities

	_control_array_size : int*
		array containing the number of control points (i.e. time slices) for each control

	_found_new_point : int*
		0 if the algorithm doesn't find a valid control point during the current iteration, 1 if it does find a valid control point to be added to the schedule.

	Returns :
	---------

	int :
		the function returns 1 if the current time point can be executed (i.e. no conflicts with other time slices), and returns 0 if the current time slice has been discarded due 
		to conflicts with time slices in the next scheduler period
		
	**/
	int point_counter;
	static_priority_scheduler_control_point tmp_control_point;

	point_counter = 0;

	// get next closest time points
	for(int ctrl_id = 0; ctrl_id < _n_controls_in_period; ctrl_id++)
	{
		if(_control_array_size[ctrl_id] > 0)
		{
			// store next closest time point id for each time array
			_closest_control_points_indices[ctrl_id] = static_priority_scheduler_get_next_control_point_id(_t, _time_array[ctrl_id], _control_array_size[ctrl_id]);

			if(point_counter == 0)
			{
				*_found_new_point = (_closest_control_points_indices[0] == -1 ? 0 : 1);

				tmp_control_point.t = _time_array[ctrl_id][_closest_control_points_indices[0]];
				tmp_control_point.t_slice = _time_slices[ctrl_id][_closest_control_points_indices[0]];
				tmp_control_point.priority = _priorities[ctrl_id][_closest_control_points_indices[0]];
				tmp_control_point.control_id = 0;
			}
			else
			{
				// store the closest time point
				if(_closest_control_points_indices[ctrl_id] != -1 
				&& ((_time_array[ctrl_id][_closest_control_points_indices[ctrl_id]] <= tmp_control_point.t || (_time_array[ctrl_id][_closest_control_points_indices[ctrl_id]] == tmp_control_point.t && _priorities[ctrl_id][_closest_control_points_indices[ctrl_id]] < tmp_control_point.priority)) || *_found_new_point != 1)) // if first point of array or if there is a closest time point
				{
					*_found_new_point = 1;
					
					tmp_control_point.t = _time_array[ctrl_id][_closest_control_points_indices[ctrl_id]];
					tmp_control_point.t_slice = _time_slices[ctrl_id][_closest_control_points_indices[ctrl_id]];
					tmp_control_point.priority = _priorities[ctrl_id][_closest_control_points_indices[ctrl_id]];
					tmp_control_point.control_id = ctrl_id;
				}
			}

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

int static_priority_scheduler_check_conflicts(
	double *_t, 
	int _n_controls_in_period, 
	static_priority_scheduler_control_point _next_control_point, 
	int* _closest_control_points_indices, 
	double **_time_array,
	double **_time_slices,
	int **_priorities,
	int *_control_array_size)
{
	/**

	function : static_priority_scheduler_check_conflicts

	Checks if there is a conflict between the current time slice to be added to the schedule and the time slices of other controls.
	In order to check for the said conflicts, the algorithm checks if one of the closest time slices of other controls with higer priorities intersect the time slice to be added.

	Parameters :
	------------
	_t : double*
		current position of the time cursor. This cursor represents the current position of the scheduling algorithme in the current scheduler period.

	_n_controls_in_period : int
		number of distinct controls to be scheduled

	_next_control_point : struct static_priority_scheduler_control_point
		this structure encapsulates the next control point to be executed. The algorithm checks if there is a time slice with higer priority to be executed during the current time slice _next_control_point

	_closest_control_points_indices

	_closest_control_points_indices : int*
		time point index of the closest time slice

	_time_array : double**
		array of control time points

	_time_slices : double**
		array of control time slice durations

	_priorities : double**
		array of control priorities

	_control_array_size : int*
		array containing the number of control points (i.e. time slices) for each control

	Returns :
	---------

	int :
		the function returns 1 if the current time point can be executed (i.e. no conflicts with other time slices), and returns 0 if the current time slice has been discarded due 
		to conflicts with time slices in the next scheduler period
		
	**/
	int iflag;

	iflag = 1;

	// check for conflicts with every other controls with higher priority
	for(int ctrl_id = 0; ctrl_id < _n_controls_in_period; ctrl_id++)
	{
		if (ctrl_id != _next_control_point.control_id && _closest_control_points_indices[ctrl_id] != -1)
		{
			// get control time array
			if(_control_array_size[ctrl_id] != -1)
			{
				if(
				_time_array[ctrl_id][_closest_control_points_indices[ctrl_id]] - _next_control_point.t < _next_control_point.t_slice 
				&& _next_control_point.priority > _priorities[ctrl_id][_closest_control_points_indices[ctrl_id]]) // if we find a new control point with higher priority inside the considered time slice
				{
					iflag = 0;
					
					// get furthest time point which is preventing control from executing 
					if (*_t < _time_array[ctrl_id][_closest_control_points_indices[ctrl_id]])
					{
						*_t = _time_array[ctrl_id][_closest_control_points_indices[ctrl_id]];
					}
				}
			}
		}
	}
	return iflag;
}

int static_priority_scheduler_get_next_control_point_id(
	double _t, 
	double* _time_array, 
	int _time_array_size)
{
	/**

	function : static_priority_scheduler_boundary

	This function looks for the time slice closest to the time cursor in the given time array and returns its index

	Parameters :
	------------
	_t : double*
		current position of the time cursor. This cursor represents the current position of the scheduling algorithme in the current scheduler period.
	
	_time_array : double* 
		time array to be checked
	
	_time_array_size : int
		size of the time array to be checked

	Returns :
	---------

	int :
		index of the closest time point in the array
		
	**/
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

void static_priority_scheduler_get_control_period_indices(
	int _scheduler_period_id, 
	int *_control_period_indices, 
	int _n_controls)
{
	/**

	function : static_priority_scheduler_boundary

	Computes the index of the time sub array which corresponds to a given scheduler period.

	The controls are splitted accordingly to the scheduler period. This ensures that there is a one to one correspondance between the control periods (splitted time array) and the 
	scheduler periods.

	Parameters :
	------------
	_scheduler_period_id : int
		current scheduler period.
	
	_control_period_indices : int*
		index of the current control period associated with the scheduler period of index _scheduler_period_id.
	
	_n_controls : int
		number of distinct controls to be scheduled.
		
	**/
	for(int ctrl_id = 0; ctrl_id < _n_controls; ctrl_id++)
		_control_period_indices[ctrl_id] = callback_static_priority_scheduler_python_functions.get_control_period_id(ctrl_id, _scheduler_period_id);
}

int static_priority_scheduler_boundary(
	double *_t,
	int _period_id,
	int _n_controls_in_period,
	static_priority_scheduler_control_point _next_control_point)
{
	/**

	function : static_priority_scheduler_boundary

	Checks if if there is a point which prevents the execution of the current time slice (i.e if there is an other time slice with higher priority executing during the current time slice)
	in the next schedule perdiod.
	This function is used as a boundary condition ensureing the continuity of the RADAR schedule between two consecutive scheduler periods.

	If there is a time slice with higer priority interrupting the next control point previously found, the algorithm will discard the latter and move the time cursor to the higher priority time slice starting.

	Parameters :
	------------
	_t : double*
		current position of the time cursor. This cursor represents the current position of the scheduling algorithme in the current scheduler period.
	
	_period_id : int 
		index of the current scheduler period
	
	_n_controls_in_period : int
		number of distinct controls to be scheduled
	
	_next_control_point : struct static_priority_scheduler_control_point
		this structure encapsulates the next control point to be executed. The algorithm checks if there is a time slice with higer priority to be executed during the current time slice _next_control_point
	
	Returns :
	---------

	int :
		the function returns 1 if the current time point can be executed (i.e. no conflicts with other time slices), and returns 0 if the current time slice has been discarded due 
		to conflicts with time slices in the next scheduler period

	**/
	int* control_period_ids;
	int next_priority;
	int found_new_point;

	double next_control_time_point;
	double next_t_slice;

	found_new_point = 1;

	// get indices of controls subarrays inside the current scheduler period
	control_period_ids = (int*)malloc(_n_controls_in_period * sizeof(int));

	static_priority_scheduler_get_control_period_indices(_period_id+1, control_period_ids, _n_controls_in_period);

	for(int ctrl_id = 0; ctrl_id < _n_controls_in_period; ctrl_id++)
	{
		if (ctrl_id != _next_control_point.control_id)
		{
			int point_exists = callback_static_priority_scheduler_python_functions.get_control_parameters(ctrl_id, control_period_ids[ctrl_id], 0, &next_control_time_point, &next_t_slice, &next_priority);

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


void get_control_sequence_time_indices(
	double *_t_final_sequence,
	double *_t_slice_final_sequence,
	double *_t_sequence_pointing_directions,
	double *_t_control,
	double *_t_dirs_control,
	int *_pdirs_control_id,
	int *_pdirs_control_t_id,
	int *_active_control_indices,
	int _control_id,
	int _n_time_points_controls,
	int _n_dirs_control,
	int _n_time_points_final_sequence,
	int _n_dirs_final_sequence,
	int _first_iteration,
	void (*_callback_save_arrays)(double*, int*, int*, int))
{
	/*
		Gets the indices of the control time points in a given control time array that correspond to the time points in the final scheduler control sequence.
	*/
	int stop_index;
	int start_index;
	int n_indices;
	int t_control_id;
	int insertion_id;

	int *new_pdir_ctrl_t_id;
	int *new_pdir_ctrl_id;
	double *new_pdir_t;

	new_pdir_ctrl_t_id = (int*)malloc(_n_dirs_final_sequence*sizeof(int));
	new_pdir_ctrl_id = (int*)malloc(_n_dirs_final_sequence*sizeof(int));
	new_pdir_t = (double*)malloc(_n_dirs_final_sequence*sizeof(double));

	for(int i = 0; i < _n_dirs_final_sequence; i++)
	{
		new_pdir_ctrl_t_id[i] = _pdirs_control_t_id[i];
		new_pdir_ctrl_id[i] = _pdirs_control_id[i];
		new_pdir_t[i] = _t_sequence_pointing_directions[i];
	}

	if(_first_iteration == 1)
	{
		_n_dirs_final_sequence = 0;
	}

	// for each time point in final control sequence
	for(int ti = 0; ti < _n_time_points_final_sequence; ti++)
	{
		t_control_id = find_time_index(_t_control, _t_final_sequence[ti], 1, 1, _n_time_points_controls);

		if (t_control_id > -1 && _active_control_indices[ti] == _control_id)
		{	
			// get start and end control time slice indices in control time array
			start_index = find_time_index(_t_dirs_control, _t_final_sequence[ti], 1, 1, _n_dirs_control);
			stop_index = start_index;

			while(_t_dirs_control[stop_index+1] <= _t_final_sequence[ti] + _t_slice_final_sequence[ti] && stop_index != -1)
			{
				if(stop_index == _n_dirs_control - 1)
				{
					stop_index = -1;
				}
				else
				{
					stop_index++;
				}
			}

			n_indices = stop_index - start_index + 1;
			//printf("t=%f, (%f, %f) - start index = %d, end index = %d, t0=%f, tf=%f\n", _t_final_sequence[ti], _t_final_sequence[0], _t_final_sequence[_n_time_points_final_sequence-1], start_index, stop_index, _t_dirs_control[0], _t_dirs_control[_n_dirs_control-1]);

			if(start_index > -1 && stop_index > -1 && n_indices > 0)
			{
				new_pdir_ctrl_t_id = (int*)realloc(new_pdir_ctrl_t_id, (_n_dirs_final_sequence + n_indices)*sizeof(int));
				new_pdir_ctrl_id = (int*)realloc(new_pdir_ctrl_id, (_n_dirs_final_sequence + n_indices)*sizeof(int));
				new_pdir_t = (double*)realloc(new_pdir_t, (_n_dirs_final_sequence + n_indices)*sizeof(double));

				// find index at which the new pointing direction has to be inserted (if _allocate_time_array is 1, then look for closest existing time point)
				insertion_id = find_time_index(_t_sequence_pointing_directions, _t_final_sequence[ti], 1, 0, _n_dirs_final_sequence);
				
				if(insertion_id == -1 || _n_dirs_final_sequence == 0)
				{
					insertion_id = 0;
				}
				else if(insertion_id == -2)
				{
					insertion_id = _n_dirs_final_sequence;
				}

				for(int j = _n_dirs_final_sequence-1; j >= insertion_id; j--)
				{
					new_pdir_ctrl_id[j + n_indices] = new_pdir_ctrl_id[j];
					new_pdir_ctrl_t_id[j + n_indices] = new_pdir_ctrl_t_id[j];
					new_pdir_t[j + n_indices] = new_pdir_t[j];
				}

				for(int index = 0; index < n_indices; index++)
				{				
					new_pdir_ctrl_id[insertion_id + index] = _control_id;
					new_pdir_ctrl_t_id[insertion_id + index] = start_index + index;
					new_pdir_t[insertion_id + index] = _t_dirs_control[start_index + index];
					
					//printf("point t=%f (ctrl_id %d) added -> id=%d\n", new_pdir_t[insertion_id + index], new_pdir_ctrl_id[insertion_id + index], start_index + index);				
				}

				_n_dirs_final_sequence += n_indices;

				/*for(int i = 0; i < _n_dirs_final_sequence; i++)
				{
					printf("new_pdir_t[%d] : %f\n", i, new_pdir_t[i]);
				}*/
			}
		}
	}
	_callback_save_arrays(new_pdir_t, new_pdir_ctrl_id, new_pdir_ctrl_t_id, _n_dirs_final_sequence);

	free(new_pdir_t);
	free(new_pdir_ctrl_t_id);
	free(new_pdir_ctrl_id);
}

int find_time_index(
	double *_t,
	double _t_target,
	int first, // if not first (first=0), we look for the last time point
	int exact, // if not exact (exact=0), we look for the closest point
	int N)
{
	int i_start, i_end, i_mid;
	int index;

	i_start = 0;
	i_end = N-1;

	if(_t_target == _t[i_end])								index = i_end;
	else if(_t_target == _t[i_start])						index = i_start;
	else if(_t_target > _t[i_end])							index = -2;
	else if(_t_target < _t[0])								index = -1;
	else
	{
		index = -3;

		while(i_end - i_start > 1)
		{
			i_mid = (int)((i_start + i_end)/2);

			if(exact == 0)
			{
				index = i_mid;
			}

			if(_t_target > _t[i_mid])
			{
				i_start = i_mid;
			}
			else if (_t_target < _t[i_mid])
			{
				i_end = i_mid;
			}
			else
			{
				i_start = i_end;
				index = i_mid;
			}

			//printf("t=%f, target = %f -> i_start %d, i_mid %d, i_end %d\n", _t[i_mid], _t_target, i_start, i_mid, i_end);
			//printf("t=%f / exact=%d\n", _t[i_end], exact);
		}
	}
	

	if(index > -1)
	{
		while(_t[index] == _t[index + 1 - 2*first] && (exact == 1))
			index +=  1 - 2*first;
	}
	//printf("return index = %d\n", index);
	return index;
}