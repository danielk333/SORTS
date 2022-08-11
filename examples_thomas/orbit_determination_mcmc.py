import numpy as np
import matplotlib.pyplot as plt
import pickle

import sorts
radar = sorts.radars.eiscat3d

def save_data(obj):
    try:
        with open("data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
 
# Profiler & logger
logger = sorts.profiling.get_logger('static')
p = sorts.profiling.Profiler()

# simulation parameters
t_end = 3600.0*10
t_slice = 0.1

# Object definition
# Propagator
Prop_cls = sorts.propagator.Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

# Object
space_object = sorts.SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7000e3, 
        e = 0.0,
        i = 78,
        raan = 86,
        aop = 0, 
        mu0 = 50,
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    )
print(space_object)

# get states and passes
t_states = sorts.equidistant_sampling(orbit=space_object.state, start_t=0, end_t=t_end*1.1, max_dpos=50e3)
object_states = space_object.get_state(t_states)
interpolator = sorts.interpolation.Legendre8(object_states, t_states)
radar_passes = radar.find_passes(t_states, object_states, cache_data=False) 
print("passes : ", radar_passes)

# controller
t = np.arange(0, t_end, t_slice)
scan_bp = sorts.scans.Beampark(azimuth=0.0, elevation=90.0, dwell=t_slice)
controller = sorts.Scanner(logger=logger, profiler=p)
controls = controller.generate_controls(t, radar, scan_bp, r=np.linspace(100e3, 3000e3, 30))
radar_states = radar.control(controls)

# observe passes
# data = radar.observe_passes(radar_passes, radar_states, space_object, snr_limit=False, parallelization=True, interpolator=interpolator, save_states=True, logger=logger, profiler=p)
# save_data(data)
data = load_data("data.pickle")

# plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure()
ax0 = fig2.add_subplot(311)
ax1 = fig2.add_subplot(312)
ax2 = fig2.add_subplot(313)

sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
plt.show()

for tx in radar.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in radar.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

ax.plot(object_states[0], object_states[1], object_states[2], '--b')

for pi in range(len(radar_passes[0][0])):
    dat = data[0][0][pi]

    if dat is not None:
        max_snr_measurements = sorts.radar.measurements.measurement.get_max_snr_measurements(dat, copy=False)["measurements"]

        detection_inds = np.where(10*np.log10(max_snr_measurements['snr']) >= 10.0)[0]

        ax.plot(object_states[0, radar_passes[0][0][pi].inds], object_states[1, radar_passes[0][0][pi].inds], object_states[2, radar_passes[0][0][pi].inds], '-r', label=f'pass-{pi}')
        ax0.plot(max_snr_measurements['t_measurements']/3600.0, max_snr_measurements['range'], '-b', label=f'pass-{pi}')
        ax1.plot(max_snr_measurements['t_measurements']/3600.0, max_snr_measurements['range_rate'], '-b', label=f'pass-{pi}')
        ax2.plot(max_snr_measurements['t_measurements']/3600.0, 10*np.log10(max_snr_measurements['snr']), '-b', label=f'pass-{pi}')

        detection_data = dict(
        	t_measurements=max_snr_measurements['t_measurements'][detection_inds],
        	snr=max_snr_measurements['snr'][detection_inds],
        	range=max_snr_measurements['range'][detection_inds],
        	range_rate=max_snr_measurements['range_rate'][detection_inds],
        )

print(p)

# Adding noise to snr and propagate linear uncertainties
sigma_snr = 5.0 # dB
radar.tx[0].n_ipp = 1
err = sorts.LinearizedCoded(radar.tx[0])

detection_data['snr'] = 10**(np.log10(detection_data['snr'])+np.random.normal(0, sigma_snr, len(detection_data['snr']))/10.0)
detection_data['range'] = err.range(detection_data['range'], detection_data['snr'])
detection_data['range_rate'] = err.range_rate(detection_data['range_rate'], detection_data['snr'])

ax0.plot(detection_data['t_measurements']/3600.0, detection_data['range'], '+r', label=f'pass-{pi}')
ax1.plot(detection_data['t_measurements']/3600.0, detection_data['range_rate'], '+r', label=f'pass-{pi}')
ax2.plot(detection_data['t_measurements']/3600.0, 10*np.log10(detection_data['snr']), '+r', label=f'pass-{pi}')

# pointing direction uncertainty
sigma_az = 0.1*np.pi/180
sigma_el = 0.15*np.pi/180

# compute uncertainties over pointing direction tx
az = radar.tx[0].beam.azimuth*np.pi/180
el = radar.tx[0].beam.elevation*np.pi/180
J = np.array([	[np.sin(el)*np.cos(az),    -np.cos(el)*np.sin(az)],
				[np.sin(el)*np.sin(az), 	np.cos(el)*np.cos(az)],
				[-np.cos(el), 				np.sin(az)]])
sigma_enu = J.dot(np.diag([sigma_el**2, sigma_az**2])).dot(J.T)
J = sorts.frames.enu_to_ecef(radar.tx[0].lat, radar.tx[0].lon, radar.tx[0].alt, np.eye(3)).reshape(3, 3)
sigma_ecef = J.dot(sigma_enu).dot(J.T)

# compute uncertainties over pointing direction rx


print(sigma_ecef)


plt.show()