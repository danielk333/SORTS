#include <stdio.h>
#include <stdlib.h>

#include <math.h>

void compute_intersection_points(double *_tx_directions, double *_rx_directions, double *_tx_ecef, double *_rx_ecef, double *_intersection_points, double _range_rtol, int *_keep, int _n_tx, int _n_rx, int _n_time_points);
void compute_barycenter(
    double *points,
    double *barycenter,
    int n_points
    );