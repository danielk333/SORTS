#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <math.h>

struct manager_python_functions_struct {
	int 	(*get_control_period_id)(int, int); //(int _index, double* _time_array, double* _t_slice, int* _priority, int* _time_array_size),
    int 	(*get_control_array_size)(int, int); //(int _index, double* _time_array, double* _t_slice, int* _priority, int* _time_array_size),
	int 	(*get_control_parameters)(int , int , int, double*, double*, int*);
	void 	(*get_control_arrays)(int, int, double* , double* , int*); //(int _index, double* _time_array, double* _t_slice, int* _priority),
	void 	(*save_new_control_arrays)(int* , int* , int);
};

struct control_point_struct {
	int  	control_id;
	int 	priority;
	double 	t;
	double 	t_slice;
};

typedef struct control_point_struct control_point;
typedef struct manager_python_functions_struct manager_python_functions;

manager_python_functions callback_manager_python_functions;

void init_manager(int(*_get_control_period_id)(int, int), int(*get_control_array_size)(int, int), int(*_get_control_parameters)(int , int , int, double*, double*, int*), void(*_get_control_arrays)(int, int, double* , double* , int*), void(*_save_new_control_arrays)(int* , int* , int));
double run_manager(int _period_id,int _n_controls_in_period, double _t_start, double _t_end);

void get_next_time_points(double _t, int _n_controls_in_period, control_point* _closest_control_point, int* _closest_control_points_indices, int* control_period_ids, int *_found_new_point);
int check_conflicts(double* _t, int _n_controls_in_period, control_point _closest_control_point, int* _closest_control_points_indices, int* control_period_ids);

int get_next_control_point_id(double _current_time_point, double* _time_array, int _time_array_size);
void get_control_period_indices(int _manager_period_id, int *_control_period_indices, int _n_controls);
int boundary(double *_t, int _period_id, int _n_controls_in_period, control_point _next_control_point);