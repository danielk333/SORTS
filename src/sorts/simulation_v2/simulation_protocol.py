import logging, typing as t
from sorts.simulation_v2.observation import Observation

logger = logging.getLogger(__name__)


class SimulationProtocol(t.Protocol):
    """Defines the top level API of a simulation object"""

    # TODO: re-eval what the return type should be
    def propagate_and_sample_space_objects_states(self) -> t.Any:
        """
        Use the sampler the get the delta time of space object within the simulation `start_time` and `end_time`
        """
        ...

    def calculate_observations(self) -> list[list[Observation]]: ...
