#!/usr/bin/env python

'''

'''

import numpy as np

from .scheduler import Scheduler

class PointingSchedule(Scheduler):
    '''#TODO: Docstring
    '''

    def generate_schedule(self, t, generator):
        
        rxp = []
        txp = []
        rx_pos = []
        tx_pos = []
        metas = []
        for ind, mrad in enumerate(generator):
            radar, meta = mrad

            metas.append(meta)
            
            rxp.append([])
            rx_pos.append([])
            for ri, rx in enumerate(radar.rx):
                rxp[-1].append(rx.pointing_ecef)
                rx_pos[-1].append(rx.ecef)
            
            txp.append([])
            tx_pos.append([])
            for ti, tx in enumerate(radar.tx):
                txp[-1].append(tx.pointing_ecef)
                tx_pos[-1].append(tx.ecef)

        data = {
            't': t,
            'rx': rxp,
            'tx': txp,
            'rx_pos': rx_pos,
            'tx_pos': tx_pos,
            'meta': metas,
        }
        return data
