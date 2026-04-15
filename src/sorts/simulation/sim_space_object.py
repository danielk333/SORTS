import dataclasses, typing as t
import numpy as np
import numpy.typing as npt
from sorts.types import Datetime64_us, EcefStates, Datetime_Like
from sorts.utils import to_datetime64_us
from sorts.interpolation import Interpolation
from sorts.propagator import Propagator
from sorts.space_object import SpaceObject


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimSpaceObject:
    """
    A richer representation of space object for use in a simulation,
    refers back to `SpaceObject` internally.
    """

    space_object: SpaceObject
    propagator: Propagator | None = None
    interpolator: t.Type[Interpolation] | None = None

    times: npt.NDArray[Datetime64_us] | None = None
    states: EcefStates | None = None
    states_interpolation: Interpolation | None = None

    def propagate(
        self,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        time_step: float,
    ) -> t.Self:
        """
        Propagate the states of a space object.

        Requires:
            - A non-`None` `self.propagator`.
        """

        if self.propagator is None:
            raise RuntimeError("`self.propagator` has to be non-`None`")

        start_time = to_datetime64_us(start_time)
        end_time = to_datetime64_us(end_time)

        dt = (end_time - start_time) / np.timedelta64(1, "s")
        t0 = (start_time - to_datetime64_us(self.space_object.epoch)) / np.timedelta64(1, "s")
        times = np.arange(t0, t0 + dt, time_step, dtype=np.float64)

        itrs_states = self.propagator.propagate(self.space_object, times)

        return dataclasses.replace(self, times=times, states=itrs_states)

    def interpolate(self, t: npt.NDArray[np.number]) -> EcefStates:
        """
        Interpolate the states of a space object.

        Requires:
            - A non-`None` `self.interpolator`.
            - A non-`None` `self.times`.
            - A non-`None` `self.states`.
        """

        if self.interpolator is None:
            raise RuntimeError("`self.interpolator` has to be non-`None`")
        if self.times is None:
            raise RuntimeError("`self.times` has to be non-`None`")
        if self.states is None:
            raise RuntimeError("`self.states` has to be non-`None`")

        interpolator = self.interpolator(self.states, self.times)
        states = interpolator.get_state(t)

        return states
