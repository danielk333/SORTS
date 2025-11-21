import abc
from sorts import radar, scheduling


class ControllerBase(abc.ABC):
    @abc.abstractmethod
    def get_experiment_detail(self) -> scheduling.ExperimentDetail: ...

    @abc.abstractmethod
    def get_experiment_id_station_id_pairs_map(
        self,
    ) -> scheduling.ExperimentIdStationIdPairsMap: ...

    @abc.abstractmethod
    def get_station_map(self) -> dict[radar.StationId, radar.Station]: ...
