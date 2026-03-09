"""
Functions that do not belong to a particular subpackage.
"""

import typing as t
from astropy.time import Time
from sorts.types import StateType
from sorts.space_object import SpaceObject
from sorts.interpolation import Interpolator
from sorts.propagator import Propagator
from sorts.interpolated_propagation import InterpolatedPropagation


class SpaceObjectInterpolatedPropagationPair(t.NamedTuple):
    space_object: SpaceObject
    interpolated_propagation: InterpolatedPropagation


def duplicate_and_perturbate_space_object(
    space_object: SpaceObject,
    propagator: Propagator,
    interpolator_class: t.Type[Interpolator],
    start_time: Time,
    end_time: Time,
    time_step: float,
    perturbation_format: StateType = "cartesian",
    pert_val: tuple[float, float, float, float, float, float] = (
        1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5  # fmt: skip
    ),
) -> list[SpaceObjectInterpolatedPropagationPair]:
    # TODO: detail structure/explanation in docstring

    # duplicate list items
    perturbed_objects: list[SpaceObjectInterpolatedPropagationPair] = []

    # perturbate all state variables and leave one original
    # i.e. len 7, [(true_spobj_list, true_prop list), (pert_spobj_prop_list, ...) ...x6]
    for idx in range(7):
        # the original spobj are left intact, the rest are copied and perturbed
        if idx == 0:
            new_obj = space_object
        else:
            new_obj = space_object.copy()

            if perturbation_format == "kepler":
                new_obj.state._kep[idx - 1, 0] += pert_val[idx - 1]
                new_obj.state.calculate_cartesian()
            elif perturbation_format == "cartesian":
                new_obj.state._cart[idx - 1, 0] += pert_val[idx - 1]
                new_obj.state.calculate_kepler()

        prop_interp = InterpolatedPropagation.from_space_object(
            space_object=new_obj,
            propagator=propagator,
            interpolator_class=interpolator_class,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step,
        )
        perturbed_objects.append(
            SpaceObjectInterpolatedPropagationPair(
                new_obj,
                prop_interp,
            )
        )
    return perturbed_objects
