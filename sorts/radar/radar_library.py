#!/usr/bin/env python

'''A collection of :class:`radar_config.RadarSystem` instances, such as EISCAT 3D and EISCAT UHF.

'''

import antenna_library as alib
import antenna
import radar_scan_library as rslib
from radar_config import RadarSystem


def eiscat_3d(beam='interp', stage=1):
    '''The EISCAT_3D system.

    For more information see:
      * `EISCAT <https://eiscat.se/>`_
      * `EISCAT 3D <https://www.eiscat.se/eiscat3d/>`_

    :param str beam: Decides what initial antenna radiation-model to use.
    :param int stage: The stage of development of EISCAT 3D. 

    **EISCAT 3D Stages:**

      * Stage 1: As of writing it is assumed to have all of the antennas in place but only transmitters on half of the antennas in a dense core ,i.e. TX will have 42 dB peak gain while RX still has 45 dB peak gain. 3 Sites will exist, one is a TX and RX, the other 2 RX sites.
      * Stage 2: Both TX and RX sites will have 45 dB peak gain.
      * Stage 3: (NOT IMPLEMENTED HERE) 2 additional RX sites will be added.


    **Beam options:**

      * gauss: Gaussian tapered beam model :func:`antenna_library.planar_beam`.
      * interp: Interpolated array pattern.
      * array: Ideal summation of all antennas in the array :func:`antenna_library.e3d_array_beam_stage1` and :func:`antenna_library.e3d_array_beam`.


    # TODO: Geographical location measured with? Probably WGS84.
    '''
    e3d_freq = 233e6

    if stage == 1:
        e3d_tx_gain = 10**4.3
        a0_tx = 20.0
    elif stage == 2:
        e3d_tx_gain = 10**4.5
        a0_tx = 40.0
    else:
        raise Exception('Stage "{}" not recognized.'.format(stage))

    e3d_rx_gain = 10**4.5
     
    if beam == 'gauss':
        tx_beam_ski = alib.planar_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_tx_gain,
            f=e3d_freq,
            a0=a0_tx,
            az1=0.0,
            el1=90.0,
        )
        rx_beam_ski = alib.planar_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
            f=e3d_freq,
            a0=40.0,
            az1=0.0,
            el1=90.0,
        )
        rx_beam_kar = alib.planar_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
            f=e3d_freq,
            a0=40.0,
            az1=0.0,
            el1=90.0,
        )
        rx_beam_kai = alib.planar_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
            f=e3d_freq,
            a0=40.0,
            az1=0.0,
            el1=90.0,
        )

    elif beam == 'array':
        if stage == 1:
            tx_beam_ski = alib.e3d_array_beam_stage1(
                az0 = 0,
                el0 = 90,
                I_0=e3d_tx_gain,
                opt='dense',
            )
        elif stage == 2:
            tx_beam_ski = alib.e3d_array_beam(
                az0 = 0,
                el0 = 90,
                I_0=e3d_tx_gain,
            )

        rx_beam_ski = alib.e3d_array_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
        )
        rx_beam_kar = alib.e3d_array_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
        )
        rx_beam_kai = alib.e3d_array_beam(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
        )
    elif beam == 'interp':
        if stage == 1:
            tx_beam_ski = alib.e3d_array_beam_stage1_dense_interp(
                az0 = 0,
                el0 = 90,
                I_0=e3d_tx_gain,
            )
        elif stage == 2:
            tx_beam_ski = alib.e3d_array_beam_interp(
                az0 = 0,
                el0 = 90,
                I_0=e3d_tx_gain,
            )

        rx_beam_ski = alib.e3d_array_beam_interp(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
        )
        rx_beam_kar = alib.e3d_array_beam_interp(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
        )
        rx_beam_kai = alib.e3d_array_beam_interp(
            az0 = 0,
            el0 = 90,
            I_0=e3d_rx_gain,
        )
    else:
        raise Exception('Beam model "{}" not recognized.'.format(beam))


    ski_lat = 69.34023844
    ski_lon = 20.313166
    ski_alt = 0.0

    ski = antenna.AntennaRX(
            name = "Skibotn",
            lat = ski_lat,
            lon = ski_lon,
            alt = ski_alt,
            el_thresh = 30,
            freq = e3d_freq,
            rx_noise = 150,
            beam = rx_beam_ski,
        )

    dwell_time = 0.1

    scan = rslib.ew_fence_model(
        lat = ski_lat,
        lon = ski_lon,
        alt = ski_alt,
        min_el = 30,
        angle_step = 1.0,
        dwell_time = dwell_time,
    )

    ski_tx = antenna.AntennaTX(
        name = "Skibotn TX",
        lat = ski_lat,
        lon = ski_lon,
        alt = ski_alt,
        el_thresh = 30,
        freq = e3d_freq,
        rx_noise = 150,
        beam = tx_beam_ski,
        scan = scan,
        tx_power = 5e6, # 5 MW
        tx_bandwidth = 100e3, # 100 kHz tx bandwidth
        duty_cycle = 0.25, # 25% duty-cycle
        pulse_length=1920e-6,
        ipp=10e-3,
        n_ipp=int(dwell_time/10e-3),
    )

    kar_lat = 68.463862
    kar_lon = 22.458859
    kar_alt = 0.0

    kar = antenna.AntennaRX(
            name = "Karesuvanto",
            lat = kar_lat,
            lon = kar_lon,
            alt = kar_alt,
            el_thresh = 30,
            freq = e3d_freq,
            rx_noise = 150,
            beam = rx_beam_kar,
        )

    kai_lat = 68.148205
    kai_lon = 19.769894
    kai_alt = 0.0

    kai = antenna.AntennaRX(
            name = "Kaiseniemi",
            lat = kai_lat,
            lon = kai_lon,
            alt = kai_alt,
            el_thresh = 30,
            freq = e3d_freq,
            rx_noise = 150,
            beam = rx_beam_kai,
        )
    # define transmit and receive antennas for a radar network.
    tx=[ski_tx]
    rx=[ski, kar, kai]

    if stage > 1:
        name = 'EISCAT 3D stage {}'.format(stage)
    else:
        name = 'EISCAT 3D'

    e3d = RadarSystem(
            tx_lst = tx, 
            rx_lst = rx,
            name = name, 
            max_on_axis=15.0, 
            min_SNRdb=1.0,
        )

    e3d.set_SNR_limits(10.0, 10.0)
    return e3d



def eiscat_3d_module(beam = 'gauss'):
    '''A single EISCAT 3D module with 100 antennas

    :param str beam: Decides what initial antenna radiation-model to use.

    **Beam options:**

      * gauss: Gaussian tapered beam model :func:`antenna_library.planar_beam`.
      * array: Ideal summation of all antennas in the array :func:`antenna_library.e3d_array_beam_stage1` and :func:`antenna_library.e3d_array_beam`.

    Based on :func:`radar_library.eiscat_3d` but with modified beam pattern.
    '''
    
    radar = eiscat_3d(beam = 'gauss')

    radar.name = 'EISCAT 3D module'

    module_gain = 10**2.2
    module_a0 = 4.0

    if beam == 'gauss':
        for tx in radar._tx:
            tx.beam.a0 = module_a0
            tx.beam.I_0 = module_gain
        for rx in radar._rx:
            rx.beam.a0 = module_a0
            rx.beam.I_0 = module_gain
    elif beam == 'array':
        for tx in radar._tx:
            tx.beam = alib.e3d_module_beam(az0=0, el0=90.0, I_0=module_gain)
        for rx in radar._rx:
            rx.beam = alib.e3d_module_beam(az0=0, el0=90.0, I_0=module_gain)
        
    return radar



def eiscat_svalbard():
    """
    The steerable antenna of the ESR radar, default settings for the Space Debris radar mode.
    """
    uhf_lat = 78.15
    uhf_lon = 16.02
    uhf_alt = 0.0

    rx_beam = alib.uhf_beam(
        az0 = 90.0, 
        el0 = 75.0, 
        I_0=10**4.25,
        f=500e6,
    )
    tx_beam = alib.uhf_beam(
        az0 = 90.0, 
        el0 = 75.0, 
        I_0=10**4.25,
        f=500e6,
    )

    scan = rslib.beampark_model(
        az = 90.0, 
        el = 75.0, 
        lat = uhf_lat, 
        lon = uhf_lon,
        alt = uhf_alt,
    )


    uhf = antenna.AntennaRX(
        name="Steerable Svalbard",
        lat=uhf_lat,
        lon=uhf_lon,
        alt=uhf_alt,
        el_thresh=30,
        freq=500e6,
        rx_noise=120,
        beam=rx_beam,
        phased = False,
        scan = scan,
    )


    uhf_tx = antenna.AntennaTX(
        name="Steerable Svalbard TX",
        lat=uhf_lat,
        lon=uhf_lon,
        alt=uhf_alt,
        el_thresh=30,
        freq=500e6,
        rx_noise=120,
        beam=tx_beam,
        scan=scan,
        tx_power=0.6e6,     # 600 kW
        tx_bandwidth=1e6,  # 1 MHz
        duty_cycle=0.125,
        pulse_length=30.0*64.0*1e-6,
        ipp=20e-3,
        n_ipp=10.0,
    )

    # EISCAT UHF beampark
    tx=[uhf_tx]
    rx=[uhf]

    euhf = RadarSystem(tx, rx, 'Eiscat Svalbard Steerable Antenna')
    euhf.set_SNR_limits(14.77,14.77)
    return euhf


def tromso_space_radar(freq = 1.2e9):
    lat = 69.5866115
    lon = 19.221555 
    alt = 85.0

    rx_beam = alib.tsr_beam(
        el0 = 90.0,
        f = freq,
    )

    scan = rslib.beampark_model(
        az = 0.0,
        el = 90.0,
        lat = lat,
        lon = lon,
        alt = alt,
    )

    tsr_rx = antenna.AntennaRX(
        name="Tromso Space Radar",
        lat = lat,
        lon = lon,
        alt = alt,
        el_thresh = 30,
        freq = freq,
        rx_noise = 100,
        beam = rx_beam,
        phased = False,
        scan = scan,
    )


    tx_beam = alib.tsr_beam(
        el0 = 90.0,
        f = freq,
    )

    tsr_tx = antenna.AntennaTX(
        name="Tromso Space Radar TX",
        lat = lat,
        lon = lon,
        alt = alt,
        el_thresh = 30,
        freq = freq,
        rx_noise = 100,
        beam = tx_beam,
        scan = scan,
        tx_power = 500.0e3,
        tx_bandwidth = 1e6,
        duty_cycle = 0.125,
        n_ipp = 10.0,
        ipp = 20e-3,
        pulse_length = 30.0*64.0*1e-6,
    )

    # EISCAT UHF beampark
    tx=[tsr_tx]
    rx=[tsr_rx]

    tsr_r = RadarSystem(tx, rx, 'Tromso Space Radar')
    tsr_r.set_SNR_limits(10.0, 10.0)
    return tsr_r
    
    

def eiscat_uhf():
    uhf_lat = 69.58649229
    uhf_lon = 19.22592538
    uhf_alt = 85.554

    rx_beam = alib.uhf_beam(
        az0 = 90.0,
        el0 = 75.0,
        I_0 = 10**4.81,
        f = 930e6
    )

    scan = rslib.beampark_model(
        az = 90.0,
        el = 75.0,
        lat = uhf_lat, 
        lon = uhf_lon,
        alt = uhf_alt,
    )

    uhf = antenna.AntennaRX(
        name="UHF Tromso",
        lat = uhf_lat,
        lon = uhf_lon,
        alt = uhf_alt,
        el_thresh = 30,
        freq = 930e6,
        rx_noise = 100,
        beam = rx_beam,
        scan = scan,
        phased = False,
    )

    tx_beam = alib.uhf_beam(
            az0 = 90.0,
            el0 = 75.0,
            I_0 = 10**4.81,
            f = 930e6
        )

    uhf_tx = antenna.AntennaTX(
        name="UHF Tromso TX",
        lat = uhf_lat,
        lon = uhf_lon,
        alt = uhf_alt,
        el_thresh = 30,
        freq = 930e6,
        rx_noise = 100,
        beam = tx_beam,
        scan = scan,
        tx_power = 1.6e6,  # 2 MW?
        tx_bandwidth = 1e6,  # 1 MHz
        duty_cycle = 0.125,
        n_ipp = 10.0,
        ipp = 20e-3,
        pulse_length = 30.0*64.0*1e-6,
        phased = False,
    )
                                

    # EISCAT UHF beampark
    tx=[uhf_tx]
    rx=[uhf]

    euhf = RadarSystem(tx, rx, 'Eiscat UHF')
    euhf.set_SNR_limits(14.0, 14.0)
    return euhf

if __name__ == "__main__":
    pass


