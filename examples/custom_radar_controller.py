#!/usr/bin/env python

'''
=======================
Custom radar controller
=======================

This example showcases the use of the standard :class:`RadarController<sorts.radar.
controllers.radar_controller.RadarController>` base class to create a simple custom 
radar controller
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import sorts
from sorts.radar.scans import Fence
eiscat3d = sorts.radars.eiscat3d

scan = Fence(azimuth=90, num=100, dwell=0.1, min_elevation=30)

class MyController(sorts.RadarController):
    '''
    '''
    def __init__(self):
        self.duty_cycle_func = lambda t, p0: (1+0.2*np.sin(t*2.0))*p0

    def compute_pointing_direction(self, controls, period_id, args):
        r = args
        t = controls.t[period_id]

        pdirs = dict()
        pdirs["tx"] = np.ndarray((len(controls.radar.tx), 1, 3, len(t)), dtype=float)
        pdirs["rx"] = np.ndarray((len(controls.radar.rx), len(controls.radar.tx), 3, len(t)), dtype=float)
        
        for txi, tx in enumerate(controls.radar.tx):
            point_ecef = controls.meta["scan"].ecef_pointing(t, tx)
            pdirs["tx"][txi, 0, :, :] = point_ecef
            point_ecef = point_ecef*r + tx.ecef[:, None]
            for rxi, rx in  enumerate(controls.radar.rx):   
                point_ecef_rxi = point_ecef - rx.ecef[:, None]
                pdirs["rx"][rxi, txi, :, :] = point_ecef_rxi/np.linalg.norm(point_ecef_rxi, axis=0)[None, :]

        return pdirs

    def generate_controls(self, t, radar, scan, r=100e3, max_points=100):
        controls = sorts.radar_controls.RadarControls(radar, self, scheduler=None, priority=0)  # the controls structure is defined as a dictionnary of subcontrols
        controls.meta["scan"] = scan
        controls.set_time_slices(t, scan.dwell(), max_points=max_points)

        # compute controls
        pdir_args = r
        controls.set_pdirs(pdir_args, cache_pdirs=True)
        sorts.radar_controller.RadarController.coh_integration(controls, radar, scan.dwell())

        for txi in range(len(radar.tx)):
            power = np.ndarray((len(t),), dtype=float)
            p0 = radar.tx[txi].power
            for ti in range(len(t)):
                power[ti] = self.duty_cycle_func(t[ti], p0)
                p0 = power[ti]

            controls.add_property_control("power", radar.tx[txi], power)
        return controls

controller = MyController()
t = np.linspace(0,10,num=100, dtype=np.float64)

controls = controller.generate_controls(t, eiscat3d, scan, r=400e3)
radar_states = eiscat3d.control(controls)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
for tx in controls.radar.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in controls.radar.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

for period_id in range(controls.n_periods):
    pdirs = controls.get_pdirs(period_id)
    print(pdirs)
    sorts.plotting.plot_beam_directions(pdirs, controls.radar, ax=ax, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)

    for tx in controls.radar.tx:
        pw = controls.get_property_control("power", tx, period_id)
        ax2.plot(controls.t[period_id], pw)

ax2.set_xlabel("$t$ [$s$]")
ax2.set_ylabel("$P_{tx}$ [$W$]")
ax2.grid()

ax.set_xlim([controls.radar.tx[0].ecef[0]-600e3, controls.radar.tx[0].ecef[0]+600e3])
ax.set_ylim([controls.radar.tx[0].ecef[1]-600e3, controls.radar.tx[0].ecef[1]+600e3])
ax.set_zlim([controls.radar.tx[0].ecef[2]-600e3, controls.radar.tx[0].ecef[2]+600e3])

plt.show()
