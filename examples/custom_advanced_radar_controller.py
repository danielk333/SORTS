#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import sorts
from sorts.radar.scans import Fence
from sorts.radar.instances import eiscat3d
from sorts.radar import RadarController


scan1 = Fence(azimuth=45.0, num=100, dwell=0.1)
scan2 = Fence(azimuth=0.0, num=20, dwell=0.2, min_elevation=60.0)

class MyController(RadarController):
    '''Takes a set of scans that are interleaved according to their cycles, i.e. one cycle each.
    The pointing direction of the radars are determined based on the currently run scan from the 
    radars first TX station, i.e. the RX stations point towards the ECEF location determined by the 
    TX scan and the given range.
    '''

    def __init__(self, radar, scans, r = 400e3):
        super().__init__(radar)
        self.scans = scans
        self.cycles = np.array([sc.cycle() for sc in scans])
        self._cy_sum = np.sum(self.cycles)
        self._csum_cycles = np.cumsum(self.cycles)
        self._norm_cycles = np.cumsum(self.cycles/self._cy_sum)
        self.r = r

    def point_radar(self, t, ind):
        point = self.scans[ind].ecef_pointing(t, self.radar.tx[0])*self.r
        point += self.radar.tx[0].ecef
        self.point_ecef(point)
        return self.radar

    def generator(self, t):
        for ti in range(len(t)):
            ind = np.argmax(np.mod(t[ti]/self._cy_sum, 1.0) < self._norm_cycles)
            t_ = np.mod(t[ti], self._cy_sum) - self._csum_cycles[ind] #relative scan epoch
            yield self.point_radar(t_, ind)


e3d = MyController(radar = eiscat3d, scans=[scan1, scan2])
t = np.linspace(0,np.sum(e3d.cycles),num=300)


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
    ln, = ax.plot([], [], [], 'g-')
    ln_rx.append(ln)


def init():
    ax.set_xlim([e3d.radar.tx[0].ecef[0]-600e3, e3d.radar.tx[0].ecef[0]+600e3])
    ax.set_ylim([e3d.radar.tx[0].ecef[1]-600e3, e3d.radar.tx[0].ecef[1]+600e3])
    ax.set_zlim([e3d.radar.tx[0].ecef[2]-600e3, e3d.radar.tx[0].ecef[2]+600e3])
    return ln_tx+ln_rx+[ttl]

def update(ind):
    radar = e3d(t[ind])
    ttl.set_text(f'Time: {t[ind]:2f} s')
    for ln, tx in zip(ln_tx, radar.tx):
        point = tx.pointing_ecef*e3d.r*1.25 + tx.ecef
        ln.set_data([tx.ecef[0], point[0]], [tx.ecef[1], point[1]])
        ln.set_3d_properties([tx.ecef[2], point[2]])
    for ln, rx in zip(ln_rx, radar.rx):
        point = rx.pointing_ecef*e3d.r*1.25 + rx.ecef
        ln.set_data([rx.ecef[0], point[0]], [rx.ecef[1], point[1]])
        ln.set_3d_properties([rx.ecef[2], point[2]])

    return ln_tx+ln_rx+[ttl]

ani = FuncAnimation(
    fig, 
    update, 
    frames=range(len(t)), 
    init_func=init, 
    blit=True,
    interval=(t.max() - t.min())/len(t)
)


plt.show()
