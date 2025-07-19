import logging, typing as t
from sorts.simulation_v2.observation import Observation

logger = logging.getLogger(__name__)


# TODO: maybe this can be removed and just allow simulation to be
#   flexible on how they should be used and what their output should be
class SimulationProtocol(t.Protocol):
    """Defines the top level API of a simulation object"""

    # TODO: re-eval what the return type should be
    def propagate_and_sample_space_objects_states(self) -> t.Any:
        """
        Use the sampler the get the delta time of space object within the simulation `start_time` and `end_time`
        """
        ...

    def calculate_observations(self) -> list[Observation]: ...
