from radardef.radar_stations import EiscatVHF
from ..radar import Radar
from ..tx_rx import TX, RX


# TODO: we should dissolve the `Radar` and `Station` classes and use `RadarStation` from `radardef`?
def gen_eiscat_vhf():
    tro_stn = EiscatVHF()

    vhf = Radar(
        [
            TX(
                lat=tro_stn.lat,
                lon=tro_stn.lon,
                alt=tro_stn.alt,
                min_elevation=tro_stn.min_elevation,
                beam=tro_stn.beam,
                beam_parameters=tro_stn.beam_parameters,
                power=tro_stn.power,
                # NOTE: these param are copied from `gen_eiscat_uhf()`
                bandwidth=1e6,
                duty_cycle=0.125,
                pulse_length=30.0 * 64.0 * 1e-6,
                ipp=20e-3,
                n_ipp=10.0,
            )
        ],
        [
            RX(
                lat=tro_stn.lat,
                lon=tro_stn.lon,
                alt=tro_stn.alt,
                min_elevation=tro_stn.min_elevation,
                noise=tro_stn.noise_temperature,
                beam=tro_stn.beam,
                beam_parameters=tro_stn.beam_parameters,
            )
        ],
        min_SNRdb=10.0,
        joint_stations=[(0, 0)],
    )
    return vhf
