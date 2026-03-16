import abc
from sorts import types, radar


class ControllerBase(abc.ABC):
    @abc.abstractmethod
    def get_experiment_detail(self) -> types.ExperimentDetail: ...

    @abc.abstractmethod
    def get_experiment_id_station_id_pairs_map(self) -> types.ExperimentIdStationIdPairsMap: ...

    @abc.abstractmethod
    def get_station_map(self) -> dict[radar.StationId, radar.Station]: ...
