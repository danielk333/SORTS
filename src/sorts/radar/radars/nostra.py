import typing as t
from radardef import RadarStation
from ..radar import Radar
from ..tx_rx import TX, RX
from ..radar_design import estimate_radar_parameters


class NostraRadarStations(t.NamedTuple):
    """
    A tuple of 6 `RadarStation`:
    `(
        se_tx, se_rx,
        no_tx, no_rx,
        fi_tx, fi_rx,
    )`.
    """

    se_tx: RadarStation
    se_rx: RadarStation
    no_tx: RadarStation
    no_rx: RadarStation
    fi_tx: RadarStation
    fi_rx: RadarStation


def gen_nostra_radar_stations(
    radar_param_estimate: dict[str, t.Any],
    frequency: float,
    duty_cycle: float,
    coherent_integration_time,
) -> NostraRadarStations:
    """
    Generate all 3 radar station of the Nostra radar system, in `radardef` style.

    Parameters:
        radar_param_estimate: The output of the function `radar_design.estimate_radar_parameters`

    Returns:
        A tuple of 6 `RadarStation`
    """

    dwell_time = coherent_integration_time / duty_cycle

    # TODO: check if we need actually need `tx_kw`, `rx_kw`
    tx_kw = dict(
        power=radar_param_estimate["tx_power"],
        bandwidth=radar_param_estimate["effective_bandwidth"],
        duty_cycle=duty_cycle,
        pulse_length=coherent_integration_time / 10,
        ipp=dwell_time / 10,
        n_ipp=10,
        min_elevation=30.0,
    )
    rx_kw = dict(
        noise=radar_param_estimate["t_noise"],
        min_elevation=30.0,
    )

    # TODO: station_ids are now hard-coded, but ideally should not be.

    se_tx = RadarStation(
        station_id="se_tx",
        transmitter=True,
        receiver=False,
        lat=65.89,
        lon=20.18,
        alt=0,
        beam=radar_param_estimate["beam"].copy(),
        beam_parameters=radar_param_estimate["beam_parameters"].copy(),
        frequency=frequency,
        power=tx_kw["power"],
        min_elevation=tx_kw["min_elevation"],
    )

    se_rx = RadarStation(
        station_id="se_rx",
        transmitter=False,
        receiver=True,
        lat=65.89,
        lon=20.18,
        alt=0,
        beam=radar_param_estimate["beam"].copy(),
        beam_parameters=radar_param_estimate["beam_parameters"].copy(),
        frequency=frequency,
        power=tx_kw["power"],
        min_elevation=rx_kw["min_elevation"],
    )

    no_tx = RadarStation(
        station_id="no_tx",
        transmitter=True,
        receiver=False,
        lat=68.96,
        lon=18.135,
        alt=0,
        beam=radar_param_estimate["beam"].copy(),
        beam_parameters=radar_param_estimate["beam_parameters"].copy(),
        frequency=frequency,
        power=tx_kw["power"],
        min_elevation=tx_kw["min_elevation"],
    )

    no_rx = RadarStation(
        station_id="no_rx",
        transmitter=False,
        receiver=True,
        lat=68.96,
        lon=18.135,
        alt=0,
        beam=radar_param_estimate["beam"].copy(),
        beam_parameters=radar_param_estimate["beam_parameters"].copy(),
        frequency=frequency,
        power=tx_kw["power"],
        min_elevation=rx_kw["min_elevation"],
    )

    fi_tx = RadarStation(
        station_id="fi_tx",
        transmitter=True,
        receiver=False,
        lat=67.80,
        lon=27.684,
        alt=0,
        beam=radar_param_estimate["beam"].copy(),
        beam_parameters=radar_param_estimate["beam_parameters"].copy(),
        frequency=frequency,
        power=tx_kw["power"],
        min_elevation=tx_kw["min_elevation"],
    )

    fi_rx = RadarStation(
        station_id="fi_rx",
        transmitter=False,
        receiver=True,
        lat=67.80,
        lon=27.684,
        alt=0,
        beam=radar_param_estimate["beam"].copy(),
        beam_parameters=radar_param_estimate["beam_parameters"].copy(),
        frequency=frequency,
        power=tx_kw["power"],
        min_elevation=rx_kw["min_elevation"],
    )

    return NostraRadarStations(
        se_tx=se_tx,
        se_rx=se_rx,
        no_tx=no_tx,
        no_rx=no_rx,
        fi_tx=fi_tx,
        fi_rx=fi_rx,
    )


# TODO: we should dissolve the `Radar` and `Station` classes and use `RadarStation` from `radardef`?
def gen_nostra(
    frequency,
    antenna_num,
    antenna_spacing_lambda,
    antenna_efficiency,
    antenna_input_power,
    thermal_load,
    noise_figure_db,
    amplifier_gain_db,
    insertion_loss_db,
    duty_cycle,
    t_sky,
    coherent_integration_time,
    bandwidth_limit_ratio,
):
    """The NOSTRA system."""
    data = estimate_radar_parameters(
        frequency=frequency,
        antenna_num=antenna_num,
        antenna_spacing_lambda=antenna_spacing_lambda,
        antenna_efficiency=antenna_efficiency,
        antenna_input_power=antenna_input_power,
        thermal_load=thermal_load,
        noise_figure_db=noise_figure_db,
        amplifier_gain_db=amplifier_gain_db,
        insertion_loss_db=insertion_loss_db,
        computation_power_draw_scaling=1,
        t_sky=t_sky,
        duty_cycle=duty_cycle,
        reference_snr_db=1,
        reference_ranges=[1000e3],
        coherent_integration_time=coherent_integration_time,
        bandwidth_limit_ratio=bandwidth_limit_ratio,
    )

    dwell_time = coherent_integration_time / duty_cycle
    tx_kw = dict(
        power=data["tx_power"],
        bandwidth=data["effective_bandwidth"],
        duty_cycle=duty_cycle,
        pulse_length=coherent_integration_time / 10,
        ipp=dwell_time / 10,
        n_ipp=10,
        min_elevation=30.0,
    )
    rx_kw = dict(
        noise=data["t_noise"],
        min_elevation=30.0,
    )

    stns = gen_nostra_radar_stations(
        radar_param_estimate=data,
        frequency=frequency,
        duty_cycle=duty_cycle,
        coherent_integration_time=coherent_integration_time,
    )

    se_rx = RX(
        lat=stns.se_rx.lat,
        lon=stns.se_rx.lon,
        alt=stns.se_rx.alt,
        beam=stns.se_rx.beam,
        beam_parameters=stns.se_rx.beam_parameters,
        frequency=stns.se_rx.frequency,
        **rx_kw,
    )
    se_tx = TX(
        lat=stns.se_tx.lat,
        lon=stns.se_tx.lon,
        alt=stns.se_tx.alt,
        beam=stns.se_tx.beam,
        beam_parameters=stns.se_tx.beam_parameters,
        frequency=stns.se_tx.frequency,
        **tx_kw,
    )

    no_rx = RX(
        lat=stns.no_rx.lat,
        lon=stns.no_rx.lon,
        alt=stns.no_rx.alt,
        beam=stns.no_rx.beam,
        beam_parameters=stns.no_rx.beam_parameters,
        frequency=stns.no_rx.frequency,
        **rx_kw,
    )
    no_tx = TX(
        lat=stns.no_tx.lat,
        lon=stns.no_tx.lon,
        alt=stns.no_tx.alt,
        beam=stns.no_tx.beam,
        beam_parameters=stns.no_tx.beam_parameters,
        frequency=stns.no_tx.frequency,
        **tx_kw,
    )

    fi_rx = RX(
        lat=stns.fi_rx.lat,
        lon=stns.fi_rx.lon,
        alt=stns.fi_rx.alt,
        beam=stns.fi_rx.beam,
        beam_parameters=stns.fi_rx.beam_parameters,
        frequency=stns.fi_rx.frequency,
        **rx_kw,
    )
    fi_tx = TX(
        lat=stns.fi_tx.lat,
        lon=stns.fi_tx.lon,
        alt=stns.fi_tx.alt,
        beam=stns.fi_tx.beam,
        beam_parameters=stns.fi_tx.beam_parameters,
        frequency=stns.fi_tx.frequency,
        **tx_kw,
    )
    # define transmit and receive antennas for a radar network.
    tx = [se_tx, no_tx, fi_tx]
    rx = [se_rx, no_rx, fi_rx]

    nostra = Radar(
        tx=tx,
        rx=rx,
        min_SNRdb=12.0,
        joint_stations=[(0, 0), (1, 1), (2, 2)],
    )
    return nostra
