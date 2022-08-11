#!/usr/bin/env python

'''
================================
Custom advanced radar controller
================================

TODO
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import sorts
from sorts.radar.scans import Fence
import sorts
eiscat3d = sorts.radars.eiscat3d
from sorts import RadarController


scan1 = Fence(azimuth=90, num=100, dwell=0.1)
scan2 = Fence(azimuth=0.0, num=100, dwell=0.1, min_elevation=75.0)
scans = [scan1, scan2]

class MyController(RadarController):
    '''Takes a set of scans that are interleaved according to their cycles, i.e. one cycle each.
    The pointing direction of the radars are determined based on the currently run scan from the 
    radars first TX station, i.e. the RX stations point towards the ECEF location determined by the 
    TX scan and the given range.
    '''
    def __init__(self):
        pass

    def generate_controls(self, t, radar, scans, r=np.linspace(300e3,1000e3,num=10), max_points=100):
        cycles         = np.array([sc.cycle() for sc in scans])
        cy_sum         = np.sum(cycles)
        norm_cycles    = np.cumsum(cycles/cy_sum)
        csum_cycles    = np.cumsum(cycles)

        controls = sorts.radar_controls.RadarControls(radar, self, scheduler=None, priority=0)  # the controls structure is defined as a dictionnary of subcontrols
        controls.meta["scans"] = scans
        controls.meta["r"] = r

        controls.set_time_slices(t, scans[0].dwell(), max_points=max_points)

        # compute controls
        pdir_args = (r, cycles, cy_sum, norm_cycles, csum_cycles)
        controls.set_pdirs(pdir_args, cache_pdirs=True)
        return controls


    def compute_pointing_direction(self, controls, period_id, args):
        r, cycles, cy_sum, norm_cycles, csum_cycles = args

        # set up computations
        points = np.ndarray((len(controls.radar.tx), 3, len(controls.t[period_id])), dtype=float)
        pointing_direction = dict()
        tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
        rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)

        for ind, scan in enumerate(controls.meta["scans"]):
            t_mod_cy_sum = np.mod(controls.t[period_id]/cy_sum, 1.0)
            mask = t_mod_cy_sum < norm_cycles[ind]
            if ind > 0:
                mask = np.logical_and(mask, t_mod_cy_sum > norm_cycles[ind-1])

            t_ = np.mod(controls.t[period_id][mask], cy_sum) - csum_cycles[ind] #relative scan epoch
            
            for txi in range(len(controls.radar.tx)):
                points_tmp = scan.ecef_pointing(controls.t[period_id][mask], controls.radar.tx[txi])
                for i in range(3):
                    points[txi, i, mask] = points_tmp[i, :] # get ecef pointing directions

            controls.t_slice[period_id][mask] = scan.dwell() # update time slices

        pointing_direction['tx']    = np.repeat(points[:, None, :, :], len(r), axis=3) # the beam directions are given as unit vectors in the ecef frame of reference
        point_rx_to_tx              = np.repeat(points[None, :, :, :], len(r), axis=3)*np.tile(r, len(points[0, 0]))[None, None, None, :] + tx_ecef[None, :, :, None] # compute the target points for the Rx stations
        point_rx                    = np.repeat(point_rx_to_tx, len(controls.radar.rx), axis=0) 
        rx_dirs                     = point_rx - rx_ecef[:, None, :, None]
    
        pointing_direction['rx']    = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :] # the beam directions are given as unit vectors in the ecef frame of reference
        pointing_direction['t']     = np.repeat(controls.t[period_id], len(r))

        return pointing_direction


controller = MyController()
cycles = np.array([sc.cycle() for sc in scans])
t = np.linspace(0, np.sum(cycles)*2,num=300)

controls = controller.generate_controls(t, eiscat3d, scans)

# radar = e3d(t=0)
# print(radar.rx[0].pointing)
# exit()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

sorts.plotting.grid_earth(ax)
for tx in controls.radar.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in controls.radar.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

ttl = ax.set_title('', animated=True)

ln_tx = []
ln_rx = []

for tx in controls.radar.tx:
    ln_txi, = ax.plot([], [], [], 'm-')
    ln_tx.append(ln_txi)

for rx in controls.radar.rx:
    ln_rx.append([])

    for ri in range(len(controls.meta["r"])):
        ln_rxi, = ax.plot([], [], [], 'g-')
        ln_rx[-1].append(ln_rxi)

dr = 1000e3

def init():
    ax.set_xlim([controls.radar.tx[0].ecef[0]-dr, controls.radar.tx[0].ecef[0]+dr])
    ax.set_ylim([controls.radar.tx[0].ecef[1]-dr, controls.radar.tx[0].ecef[1]+dr])
    ax.set_zlim([controls.radar.tx[0].ecef[2]-dr, controls.radar.tx[0].ecef[2]+dr])

    ax.view_init(elev=25, azim=120)

    lst = [ttl]
    for ln_rxi in ln_rx:
        lst += ln_rxi
    lst += ln_tx
    return lst

def update(ind):
    period_id = int(ind/controls.max_points)
    ti = ind%controls.max_points
    n_r = len(controls.meta["r"])

    pdirs = controls.get_pdirs(period_id)

    ax.view_init(elev=25, azim=120.0 + ind*0.2)

    ttl.set_text(f'Time: {controls.t[period_id][ti]:2f} s')
    for txi, tx in enumerate(controls.radar.tx):
        point_tx = pdirs["tx"][txi, 0, :, ti*n_r]*controls.meta["r"].max() + tx.ecef
        ln_txi = ln_tx[txi]
        ln_txi.set_data([tx.ecef[0], point_tx[0]], [tx.ecef[1], point_tx[1]])
        ln_txi.set_3d_properties([tx.ecef[2], point_tx[2]])

        for rxi, rx in enumerate(controls.radar.rx):
            ln_rxi = ln_rx[rxi]
            
            for ri in range(n_r):
                point_tx_i = pdirs["tx"][txi, 0, :,  ti*n_r + ri]*controls.meta["r"][ri] + tx.ecef
                point = pdirs["rx"][rxi, txi, :, ti*n_r + ri]*np.linalg.norm(rx.ecef - point_tx_i) + rx.ecef
                print(ti*n_r + ri)
                ln_rxi[ri].set_data([rx.ecef[0], point[0]], [rx.ecef[1], point[1]])
                ln_rxi[ri].set_3d_properties([rx.ecef[2], point[2]])

    lst = [ttl]
    for ln_rxi in ln_rx:
        lst += ln_rxi
    lst += ln_tx
    return lst


interval = (t.max() - t.min())/len(t)
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=len(t), 
    init_func=init, 
    blit=True,
    interval=interval,
)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Daniel Kastinen'), bitrate=1800)
ani.save('./examples/data/e3d_controller_multi_beam_test.mp4', writer=writer)
