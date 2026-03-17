import typing as t
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from tqdm import tqdm
from astropy.time import Time, TimeDelta
from sorts.space_object import SpaceObject
from sorts.types import Datetime64_us, EcefStates
from sorts.interpolation import Interpolator
from sorts.propagator import Propagator
from sorts.population import Population


@dataclass(kw_only=True)
class InterpolatedPropagation:
    # TODO: investigate if we can just sidestep most of the `datetime64` and just use `Time`?
    #
    #       comment from Hin:
    #       I actually prefer to get away from `Time`
    #       as soon as we are outside of the user facing APIs.
    #       It is because `datetime64` is what `numpy` uses then we will risk
    #       having type convertions pops up in random locations in the core computation codes
    times: npt.NDArray[Datetime64_us]
    states: EcefStates
    interpolator: Interpolator
    epoch: Datetime64_us

    @classmethod
    def from_space_objects(
        cls,
        space_objects: t.Sequence[SpaceObject] | Population,
        propagator: Propagator,
        interpolator_class: t.Type[Interpolator],
        start_time: Time,
        end_time: Time,
        time_step: float,
        progress: bool = False,
    ) -> list[t.Self]:
        prop_interps = []
        if progress:
            pbar = tqdm(desc="Propagating", total=len(space_objects))
        for spobj in space_objects:
            pint = cls.from_space_object(
                spobj,
                propagator,
                interpolator_class,
                start_time,
                end_time,
                time_step,
            )
            prop_interps.append(pint)
            if progress:
                pbar.update(1)
        if progress:
            pbar.close()
        return prop_interps

    @classmethod
    def from_space_object(
        cls,
        space_object: SpaceObject,
        propagator: Propagator,
        interpolator_class: t.Type[Interpolator],
        start_time: Time,
        end_time: Time,
        time_step: float,
    ) -> t.Self:
        dt = (end_time - start_time).sec
        t0 = (start_time - space_object.epoch).sec
        t_obj = np.arange(t0, t0 + dt, time_step, dtype=np.float64)
        itrs_states = propagator.propagate(space_object, t_obj)

        interp = interpolator_class(itrs_states, t_obj)
        pint = cls(
            times=(space_object.epoch + TimeDelta(t_obj, format="sec")).datetime64,
            states=itrs_states,
            interpolator=interp,
            epoch=space_object.epoch.datetime64,
        )
        return pint

    @property
    def relative_seconds(self):
        return (self.times - self.epoch) / np.timedelta64(1, "s")
