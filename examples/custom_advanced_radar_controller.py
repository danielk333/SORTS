#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import sorts
from sorts.radar.scans import Fence
from sorts.radar.instances import eiscat3d
from sorts import RadarController


scan1 = Fence(azimuth=90, num=100, dwell=0.1)
scan2 = Fence(azimuth=0.0, num=100, dwell=0.1, min_elevation=75.0)

class MyController(RadarController):
    '''Takes a set of scans that are interleaved according to their cycles, i.e. one cycle each.
    The pointing direction of the radars are determined based on the currently run scan from the 
    radars first TX station, i.e. the RX stations point towards the ECEF location determined by the 
    TX scan and the given range.
    '''

    def __init__(self, radar, scans, r=np.linspace(300e3,1000e3,num=10)):
        super().__init__(radar)
        self.scans = scans
        self.cycles = np.array([sc.cycle() for sc in scans])
        self._cy_sum = np.sum(self.cycles)
        self._csum_cycles = np.cumsum(self.cycles)
        self._norm_cycles = np.cumsum(self.cycles/self._cy_sum)
        self.r = r

    def point_radar(self, t, ind):
        point = self.scans[ind].ecef_pointing(t, self.radar.tx[0])

        point_tx = point + self.radar.tx[0].ecef
        point_rx = point[:,None]*self.r[None,:] + self.radar.tx[0].ecef[:,None]
        
        self.point_tx_ecef(point_tx)
        self.point_rx_ecef(point_rx)

        return self.radar

    def generator(self, t):
        for ti in range(len(t)):
            ind = np.argmax(np.mod(t[ti]/self._cy_sum, 1.0) < self._norm_cycles)
            t_ = np.mod(t[ti], self._cy_sum) - self._csum_cycles[ind] #relative scan epoch
            yield self.point_radar(t_, ind)


e3d = MyController(radar = eiscat3d, scans=[scan1, scan2])
t = np.linspace(0,np.sum(e3d.cycles)*2,num=300)


# radar = e3d(t=0)
# print(radar.rx[0].pointing)
# exit()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

sorts.plotting.grid_earth(ax)
for tx in e3d.radar.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in e3d.radar.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

ttl = ax.set_title('',animated=True)

ln_tx = []
ln_rx = []

for tx in e3d.radar.tx:
    ln, = ax.plot([], [], [], 'm-')
    ln_tx.append(ln)
for rx in e3d.radar.rx:
    ln_rx.append([])
    for ri in range(len(e3d.r)):
        ln, = ax.plot([], [], [], 'g-')
        ln_rx[-1].append(ln)



dr = 1000e3

def init():
    ax.set_xlim([e3d.radar.tx[0].ecef[0]-dr, e3d.radar.tx[0].ecef[0]+dr])
    ax.set_ylim([e3d.radar.tx[0].ecef[1]-dr, e3d.radar.tx[0].ecef[1]+dr])
    ax.set_zlim([e3d.radar.tx[0].ecef[2]-dr, e3d.radar.tx[0].ecef[2]+dr])
    ax.view_init(elev=20, azim=120)
    lst = ln_tx+[ttl]
    for ln in ln_rx:
        lst += ln
    return lst

def update(ind):
    radar = e3d(t[ind])

    ax.view_init(elev=20, azim=120.0 + ind*0.2)

    ttl.set_text(f'Time: {t[ind]:2f} s')
    for ln, tx in zip(ln_tx, radar.tx):
        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*e3d.r.max() + tx.ecef
        ln.set_data([tx.ecef[0], point_tx[0]], [tx.ecef[1], point_tx[1]])
        ln.set_3d_properties([tx.ecef[2], point_tx[2]])

        for ln, rx in zip(ln_rx, radar.rx):
            pecef = rx.pointing_ecef/np.linalg.norm(rx.pointing_ecef, axis=0)
            for ri in range(len(e3d.r)):
                point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*e3d.r[ri] + tx.ecef

                point = pecef[:,ri]*np.linalg.norm(rx.ecef - point_tx) + rx.ecef
                ln[ri].set_data([rx.ecef[0], point[0]], [rx.ecef[1], point[1]])
                ln[ri].set_3d_properties([rx.ecef[2], point[2]])
    lst = ln_tx+[ttl]
    for ln in ln_rx:
        lst += ln
    return lst + [ax]


interval = (t.max() - t.min())/len(t)
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=range(len(t)), 
    init_func=init, 
    blit=True,
    interval=interval,
)

# Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Daniel Kastinen'), bitrate=1800)
# ani.save('/home/danielk/e3d_controller_multi_beam_test.mp4', writer=writer)

plt.show()
