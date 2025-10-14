import abc
from sorts import radar, schedule


class ControllerBase(abc.ABC):
    @abc.abstractmethod
    def get_experiment_detail(self) -> schedule.ExperimentDetail: ...

    @abc.abstractmethod
    def get_station_map(self) -> dict[radar.StationId, radar.Station]: ...

    @abc.abstractmethod
    def get_station_pairs(self) -> list[tuple[radar.StationId, radar.StationId]]: ...
