#!/usr/bin/env python

'''Estimating orbit determination errors

'''
import pathlib

import numpy as np

from .linearized_coded import LinearizedCodedIonospheric
from .linearized_coded import LinearizedCoded



def orbit_determination_covariance(
        passes, 
        scheduler, 
        space_object,
        variables = ['x','y','z','vx','vy','vz','A'],
        deltas = [1e-4]*3 + [1e-6]*3 + [1e-2],
        cache_folder=None, 
        ray_bending=True,
        prior_cov_inv=None,
        transforms = {
            'A': (lambda A: np.log10(A), lambda Ainv: 10.0**Ainv),
        },
    ):
    '''Takes a series of passes and calculates a orbit determination covariance of the measurement model is linear and linearized coded errors with optionally ionospheric effects are included.
    '''

    if cache_folder is not None:
        cache_folder = pathlib.Path(cache_folder)

    #observe all the passes, including measurement Jacobian
    datas = []
    J = None
    for ind, ps in enumerate(passes):
        txi, rxi = ps.station_id

        #Now we load the error model
        if ray_bending:
            err = LinearizedCodedIonospheric(scheduler.radar.tx[txi], cache_folder=cache_folder)
        else:
            err = LinearizedCoded(scheduler.radar.tx[txi], cache_folder=cache_folder)

        #the Jacobean is stacked as [r_measurements, v_measurements]^T so we stack the measurement covariance equally
        data, J_rx = scheduler.calculate_observation_jacobian(
            ps, 
            space_object=space_object, 
            variables=variables, 
            deltas=deltas, 
            snr_limit=True,
            transforms = transforms,
        )
        if data is None:
            continue

        datas += [data]

        #now we get the expected standard deviations
        if ray_bending:
            r_stds_tx = err.range_std(data['range'], data['snr'])
        else:
            r_stds_tx = err.range_std(data['snr'])
        v_stds_tx = err.range_rate_std(data['snr'])

        #Assume uncorrelated errors = diagonal covariance matrix
        Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

        #we simply append the results on top of each other for each station
        if J is not None:
            J = np.append(J, J_rx, axis=0)
            Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
        else:
            J = J_rx
            Sigma_m_diag = Sigma_m_diag_tx

    #This means that no passes were observable
    if J is None:
        return None, None

    #diagonal matrix inverse is just element wise inverse of the diagonal
    Sigma_m_inv = np.diag(1.0/Sigma_m_diag)

    #For a thorough derivation of this formula:
    #see Fisher Information Matrix of a MLE with Gaussian errors and a Linearized measurement model
    if prior_cov_inv is not None:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + prior_cov_inv)
    else:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)

    return Sigma_orb, datas





def covariance_propagation(
        space_object, 
        orbit_cov, 
        t, 
        variables, 
        samples=100, 
        perturbation_cov=None, 
        perturbed_variables=None, 
        transforms = {
            'A': (lambda A: np.log10(A), lambda Ainv: 10.0**Ainv),
        },
    ):
    '''
    Propagate error covariance in time. The time vector should start at the moment of the epoch for the covariance matrix.
    Sample mean position and velocity error.
    Optionally add additional errors.
    '''

    ecef0 = space_object.get_state(t)

    r_diff = np.zeros(len(t), dtype=np.float64)
    v_diff = np.zeros(len(t), dtype=np.float64)
    for i in range(samples):
        deltas = np.random.multivariate_normal(np.zeros(orbit_cov.shape[0]),orbit_cov)

        if perturbation_cov is not None:
            pert = np.random.multivariate_normal(np.zeros(perturbation_cov.shape[0]),perturbation_cov)

        dso = space_object.copy()
        
        update = {}
        for ind, var in enumerate(variables):
            if var in transforms:
                Tx = transforms[var][0](getattr(dso, var)) + deltas[ind]
                dx = transforms[var][1](Tx)
            else:
                dx = getattr(dso, var) + deltas[ind]

            update[var] = dx

        if perturbation_cov is not None:
            for ind, var in enumerate(perturbed_variables):
                if var in update:
                    base = update[var]
                else:
                    base = getattr(dso, var)

                if var in transforms:
                    Tx = transforms[var][0](base) + pert[ind]
                    dx = transforms[var][1](Tx)
                else:
                    dx = base + pert[ind]
            
            update[var] = dx

        dso.update(**update)

        ecef1 = dso.get_state(t)

        r_diff += np.linalg.norm(ecef1[3:,:]-ecef0[3:,:], axis=0)
        v_diff += np.linalg.norm(ecef1[3:,:]-ecef0[3:,:], axis=0)


    r_diff_stdev = r_diff/samples
    v_diff_stdev = v_diff/samples

    return r_diff_stdev, v_diff_stdev

