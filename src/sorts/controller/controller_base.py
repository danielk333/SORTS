import abc
from sorts import radar, schedule


class ControllerBase(abc.ABC):
    @abc.abstractmethod
    def get_experiment_detail(self) -> schedule.ExperimentDetail: ...

    @abc.abstractmethod
    def get_experiment_id_station_id_pairs_map(self) -> schedule.ExperimentIdStationIdPairsMap: ...

    @abc.abstractmethod
    def get_station_map(self) -> dict[radar.StationId, radar.Station]: ...
