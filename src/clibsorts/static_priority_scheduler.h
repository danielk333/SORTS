#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <math.h>

struct static_priority_scheduler_python_functions_struct {
	int 	(*get_control_period_id)(int, int); //(int _index, double* _time_array, double* _t_slice, int* _priority, int* _time_array_size),
    int 	(*get_control_array_size)(int, int); //(int _index, double* _time_array, double* _t_slice, int* _priority, int* _time_array_size),
	int 	(*get_control_parameters)(int , int , int, double*, double*, int*);
	void	(*get_control_arrays)(int, double** , double** , int**, int*);
	void 	(*save_new_control_arrays)(int* , int* , int);
};

struct static_priority_scheduler_control_point_struct {
	int  	control_id;
	int 	priority;
	double 	t;
	double 	t_slice;
};

typedef struct static_priority_scheduler_control_point_struct static_priority_scheduler_control_point;
typedef struct static_priority_scheduler_python_functions_struct static_priority_scheduler_python_functions;

static_priority_scheduler_python_functions callback_static_priority_scheduler_python_functions;

void init_static_priority_scheduler(
    int(*_get_control_period_id)(int, int),
	int(*_get_control_parameters)(int , int , int, double*, double*, int*),
	void(*_get_control_arrays)(int, double** , double** , int**, int*),
	void(*_save_new_control_arrays)(int* , int* , int));

double run_static_priority_scheduler(int _period_id,int _n_controls_in_period, double _t_start, double _t_end);

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
	);

int static_priority_scheduler_check_conflicts(
	double *_t, 
	int _n_controls_in_period, 
	static_priority_scheduler_control_point _next_control_point, 
	int* _closest_control_points_indices, 
	double **_time_array,
	double **_time_slices,
	int **_priorities,
	int *_control_array_size);

int static_priority_scheduler_get_next_control_point_id(double _current_time_point, double* _time_array, int _time_array_size);
void static_priority_scheduler_get_control_period_indices(int _scheduler_period_id, int *_control_period_indices, int _n_controls);
int static_priority_scheduler_boundary(double *_t, int _period_id, int _n_controls_in_period, static_priority_scheduler_control_point _next_control_point);

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
	void (*_callback_save_arrays)(double*, int*, int*, int));

int find_time_index(
	double *_t,
	double _t_target,
	int first, // if not first (first=0), we look for the last time point
	int exact, // if not exact (exact=0), we look for the closest point
	int N);