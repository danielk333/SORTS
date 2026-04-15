import typing as t
from astropy.time import Time
from sorts import types
from sorts.space_object import SpaceObject
from sorts.interpolation import Interpolation
from sorts.propagator import Propagator
from sorts.interpolated_propagation import InterpolatedPropagation


class SpaceObjectInterpolatedPropagationPair(t.NamedTuple):
    space_object: SpaceObject
    interpolated_propagation: InterpolatedPropagation


def duplicate_and_perturbate_space_object(
    space_object: SpaceObject,
    propagator: Propagator,
    interpolator_class: t.Type[Interpolation],
    start_time: Time,
    end_time: Time,
    time_step: float,
    perturbation_format: types.StateType = "cartesian",
    pert_val: tuple[float, float, float, float, float, float] = (
        1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5  # fmt: skip
    ),
) -> types.Tuple_7[SpaceObjectInterpolatedPropagationPair]:
    """
    Perturbate the input space object by the specified method and values.

    Returns:
        A tuple of 7 `SpaceObjectInterpolatedPropagationPair`,
        where the first one is for the true space object, and the reset follows the order of input perturbation value order:
        ```
        [(true_spobj, true_prop), (pert_spobj, pert_spobj_prop) ...x6]
        ```
    """

    # duplicate list items
    spobj_interp_prop_pairs: list[SpaceObjectInterpolatedPropagationPair] = []

    # perturbate all state variables and leave one original
    # i.e. len 7, [(true_spobj, true_prop), (pert_spobj, pert_spobj_prop) ...x6]
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

        prop_interp = InterpolatedPropagation.from_space_object(
            space_object=new_obj,
            propagator=propagator,
            interpolator_class=interpolator_class,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step,
        )
        spobj_interp_prop_pairs.append(
            SpaceObjectInterpolatedPropagationPair(
                new_obj,
                prop_interp,
            )
        )

    return (
        spobj_interp_prop_pairs[0],
        spobj_interp_prop_pairs[1],
        spobj_interp_prop_pairs[2],
        spobj_interp_prop_pairs[3],
        spobj_interp_prop_pairs[4],
        spobj_interp_prop_pairs[5],
        spobj_interp_prop_pairs[6],
    )
