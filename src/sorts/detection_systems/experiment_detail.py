from dataclasses import dataclass


@dataclass
class ExperimentDetail:
    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float
