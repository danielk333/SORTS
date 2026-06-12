import dataclasses, typing as t
import numpy as np
import numpy.typing as npt
from spacecoords.interpolation import Interpolator as Interpolation
from sorts import types
from sorts.types import EcefStates, Datetime_Like, Float64_as_sec
from sorts.utils import to_datetime64_us
from sorts.propagator import Propagator
from sorts.space_object import SpaceObject


@dataclasses.dataclass(kw_only=True, frozen=True)
class SpaceObjectSimulator:
    """A convenience object for simulation of a `SpaceObject`."""

    propagator: Propagator
    interpolator: t.Type[Interpolation]

    def propagate(
        self,
        space_object: SpaceObject,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        time_step: float,
    ) -> tuple[npt.NDArray[Float64_as_sec], EcefStates]:
        """Propagate the states of a space object."""

        start_time = to_datetime64_us(start_time)
        end_time = to_datetime64_us(end_time)

        dt = (end_time - start_time) / np.timedelta64(1, "s")
        t0 = (start_time - to_datetime64_us(space_object.epoch)) / np.timedelta64(1, "s")
        times = np.arange(t0, t0 + dt, time_step, dtype=np.float64)

        itrs_states = self.propagator.propagate(space_object, times)

        return times, itrs_states

    def make_interpolation(
        self,
        times: npt.NDArray[Float64_as_sec],
        states: EcefStates,
    ) -> Interpolation:
        return self.interpolator(states=states, t=times)

    @staticmethod
    def perturbate(
        space_object: SpaceObject,
        perturbation_format: types.StateType = "cartesian",
        pert_val: tuple[float, float, float, float, float, float] = (
            1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5  # fmt: skip
        ),
    ):
        """
        Perturbate the input space object by the specified method and values.

        Returns:
            A tuple of 7 `SpaceObject`,
            where the first one is for the true space object, and the reset follows the order of input perturbation value order:
            ```
            [true_spobj, pert_spobj ...x6]
            ```
        """

        space_object_ls: list[SpaceObject] = []
        for idx in range(7):
            # the original spobj are left intact, the rest are copied and perturbed
            if idx == 0:
                new_obj = space_object
            else:
                new_obj = space_object.copy()

                if perturbation_format == "kepler":
                    new_obj.orbit._kep[idx - 1, 0] += pert_val[idx - 1]
                    new_obj.orbit.calculate_cartesian()
                elif perturbation_format == "cartesian":
                    new_obj.orbit._cart[idx - 1, 0] += pert_val[idx - 1]
                    new_obj.orbit.calculate_kepler()

            space_object_ls.append(new_obj)

        return (
            space_object_ls[0],
            space_object_ls[1],
            space_object_ls[2],
            space_object_ls[3],
            space_object_ls[4],
            space_object_ls[5],
            space_object_ls[6],
        )
