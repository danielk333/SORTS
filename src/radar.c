#include "radar.h"

void compute_intersection_points(
    double *_tx_directions,
    double *_rx_directions,
    double *_tx_ecef,
    double *_rx_ecef,
    double *_intersection_points,
    double _range_rtol,
    int *_keep,
    int _n_tx,
    int _n_rx,
    int _n_time_points)
{
    double lambda;
    double scalar_product_txrx_dir;
    double det;

    double* tmp_intersection_points = (double*)malloc(1*sizeof(double));
    double* rx_to_tx_dir            = (double*)malloc(3*sizeof(double));
    double* intersection_barycenter = (double*)malloc(3*sizeof(double));

    int tx_start_index;
    int rx_start_index;
    int tmp_intersection_points_size;

    // for each time point
    for(int ti = 0; ti < _n_time_points; ti++)
    {
        _keep[ti] = 0;

        for(int txi = 0; txi < _n_tx; txi++)
        {
            tmp_intersection_points_size = 0;

            for(int rxi = 0; rxi < _n_rx; rxi++)
            {
                tx_start_index = (int)(txi * 3 * _n_time_points);
                rx_start_index = (int)(rxi * 3 * _n_time_points * _n_tx + txi * 3 * _n_time_points);


                // compute unit vector from station txi to rxi
                for(int k = 0; k < 3; k++)
                    rx_to_tx_dir[k] = _tx_ecef[txi*3 + k] - _rx_ecef[rxi*3 + k];

                scalar_product_txrx_dir = 0.0;
                for(int k = 0; k < 3; k++)
                    scalar_product_txrx_dir += _tx_directions[tx_start_index + k * _n_time_points + ti] * _rx_directions[rx_start_index + k * _n_time_points + ti];
                               
                // the rx and tx station aren't the same (i.e. unit pointing vectors not aligned)
                if(fabs(scalar_product_txrx_dir) < 1 - 1e-5)
                {
                    // compute determinant of the matrix [ktx_i, krx_j, txi_to_rxj] to check if the 3 vectors are in the same plane
                    det = 0;
                    for(int k = 0; k < 3; k++)
                        det += rx_to_tx_dir[k] * (_tx_directions[tx_start_index + (k+1)%3 * _n_time_points + ti] * _rx_directions[rx_start_index + (k+2)%3 * _n_time_points + ti] - _tx_directions[tx_start_index + (k+2)%3 * _n_time_points + ti] * _rx_directions[rx_start_index + (k+1)%3 * _n_time_points + ti]);
                    
                    if(fabs(det) <= 1e-5) // if the 3 vectors are in the same plane then the intersection point exists
                    {
                        // compute distance from tx station to ecef point
                        lambda = 0.0;
                        for(int k = 0; k < 3; k++)
                            lambda += rx_to_tx_dir[k] * (_rx_directions[rx_start_index + k*_n_time_points + ti]*scalar_product_txrx_dir - _tx_directions[tx_start_index + k*_n_time_points + ti]);
                        
                        lambda = lambda/(1.0 - pow(scalar_product_txrx_dir, 2.0));

                        tmp_intersection_points_size += 3;
                        tmp_intersection_points = (double*)realloc(tmp_intersection_points, tmp_intersection_points_size*sizeof(double));
                        
                        // computing intersection point ecef coordinates.
                        for(int k = 0; k < 3; k++)
                        {
                            tmp_intersection_points[tmp_intersection_points_size-3 + k] = _tx_ecef[txi*3 + k] + lambda * _tx_directions[tx_start_index + k*_n_time_points + ti];
                        }

                        _keep[ti] = 1;
                    }
                    else
                    {
                        _keep[ti] = 0;
                        break;
                    }
                }
            }

            if(_keep[ti] == 1)
            {
                double range_from_barycenter;
                double range;
                int size;

                size = (int)(tmp_intersection_points_size/3);

                compute_barycenter(tmp_intersection_points, intersection_barycenter, size);

                for(int rxi = 0; rxi < size; rxi++)
                {
                    range = 0;
                    range_from_barycenter = 0;

                    for(int k = 0; k < 3; k++)
                    {
                        range_from_barycenter   += pow(intersection_barycenter[k] - tmp_intersection_points[rxi*3 + k], 2.0);
                        range                   += pow(tmp_intersection_points[rxi*3 + k], 2.0);
                    }   

                    if (sqrt(range_from_barycenter/range) < _range_rtol)
                    {
                        for(int k = 0; k < 3; k++)
                            _intersection_points[k*_n_time_points + ti] = intersection_barycenter[k];
                    }
                    else
                    {
                        _keep[ti] = 0;
                        break;
                    }
                }
            }
        }
    }

    free(intersection_barycenter);
    free(rx_to_tx_dir);

    free(tmp_intersection_points);
}

void compute_barycenter(
    double *points,
    double *barycenter,
    int n_points
    )
{
    for(int k = 0; k < 3; k++)
    {
        barycenter[k] = 0;

        for(int pi = 0; pi < n_points; pi++)
            barycenter[k] += points[pi*3 + k];

        barycenter[k] = barycenter[k]/n_points;
    }
}