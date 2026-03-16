#!/usr/bin/env python

""" """
import numpy as np
import pyant

from ..radar import Radar
from ..tx_rx import TX, RX
from ..radar_design import estimate_radar_parameters


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

    se_rx = RX(
        lat=65.89,
        lon=20.18,
        alt=0,
        beam=data["beam"].copy(),
        beam_parameters=data["beam_parameters"].copy(),
        frequency=frequency,
        **rx_kw,
    )
    se_tx = TX(
        lat=65.89,
        lon=20.18,
        alt=0,
        beam=data["beam"].copy(),
        beam_parameters=data["beam_parameters"].copy(),
        frequency=frequency,
        **tx_kw,
    )

    no_rx = RX(
        lat=68.96,
        lon=18.135,
        alt=0,
        beam=data["beam"].copy(),
        beam_parameters=data["beam_parameters"].copy(),
        frequency=frequency,
        **rx_kw,
    )
    no_tx = TX(
        lat=68.96,
        lon=18.135,
        alt=0,
        beam=data["beam"].copy(),
        beam_parameters=data["beam_parameters"].copy(),
        frequency=frequency,
        **tx_kw,
    )

    fi_rx = RX(
        lat=67.80,
        lon=27.684,
        alt=0,
        beam=data["beam"].copy(),
        beam_parameters=data["beam_parameters"].copy(),
        frequency=frequency,
        **rx_kw,
    )
    fi_tx = TX(
        lat=67.80,
        lon=27.684,
        alt=0,
        beam=data["beam"].copy(),
        beam_parameters=data["beam_parameters"].copy(),
        frequency=frequency,
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
