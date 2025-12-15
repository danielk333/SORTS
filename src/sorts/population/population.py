#!/usr/bin/env python

"""Defines a population of space objects in the form of a class."""
from pathlib import Path
from typing import Self, Callable, Any
import copy
from collections import defaultdict, OrderedDict
from functools import reduce

import h5py
import numpy as np
import numpy.typing as npt
import pyorb
from tabulate import tabulate
from astropy.time import Time

from sorts.types import StateType, NDArray_6xN, NDArray_N, IndexLike, AnomalyType, Frames
from sorts.space_object import SpaceObject


class Population:
    """Encapsulates a population of space objects as an array and functions
    for returning instances of space objects.

    The columns represent all components needed to instantiate a space object, i.e. a state (pyorb),
    properties, epoch, and identifiers.
    """

    def __init__(
        self,
        states: NDArray_6xN,
        epochs: Time,
        frame: Frames,
        parameters: dict[str, NDArray_N],
        object_ids: NDArray_N | None,
        state_format: StateType = "kepler",
        anomly_type: AnomalyType = "mean",
        dtypes: dict[str, npt.DTypeLike] | None = None,
        default_dtype: npt.DTypeLike = np.float64,
        epoch_format: str = "mjd",
        epoch_scale: str = "utc",
        degrees: bool = True,
    ):
        self.data: npt.NDArray

        assert states.shape[1] == len(epochs)
        if state_format == "kepler":
            state_keys = pyorb.Orbit.KEPLER
        elif state_format == "cartesian":
            state_keys = pyorb.Orbit.CARTESIAN

        if dtypes is None:
            dtypes = {}

        for dt in list(dtypes.values()):
            if np.dtype(dt).char == "U":
                raise TypeError(
                    "Initialized Population cannot use the save function with"
                    "Unicode [U] numpy strings, try using ASCII [S] strings instead."
                )
        self.state_fields = state_keys
        self.property_fields = list(parameters.keys())
        self.state_format = state_format
        self.epoch_format = epoch_format
        self.epoch_scale = epoch_scale
        self.anomly_type = anomly_type
        self.degrees = degrees
        self.frame = frame

        data_keys = ["id", "epoch"] + state_keys + list(parameters.keys())
        self.dtypes = OrderedDict()
        for key in data_keys:
            dt = dtypes[key] if key in dtypes else default_dtype
            self.dtypes[key] = dt

        self.allocate(len(epochs))
        self.data["id"] = np.arange(len(epochs)) if object_ids is None else object_ids
        self.data["epoch"] = epochs.to_value(epoch_format)
        for ind, key in enumerate(state_keys):
            self.data[key] = states[ind, :]
        for key, vals in parameters.items():
            self.data[key] = vals

    def __len__(self) -> int:
        return len(self.data)

    def copy(self) -> Self:
        """Return a copy of the current Population instance."""
        pop = Population()
        raise NotImplementedError()
        pop.data = self.data.copy()
        return pop

    def delete(self, inds: IndexLike):
        """Remove the rows according to the given indices.
        Supports single index, iterable of indices and slices.
        """
        if isinstance(inds, int):
            inds = [inds]
        elif isinstance(inds, slice):
            _inds = range(self.data.shape[0])
            inds = list(_inds[inds])
        elif not (isinstance(inds, list) or isinstance(inds, np.ndarray)):
            raise Exception("Cannot delete indecies given with type {}".format(type(inds)))

        mask = np.full((self.data.shape[0],), True, dtype=bool)
        for ind in inds:
            mask[ind] = False
        self.data = self.data[mask]

    def filter(self, col: str, fun: Callable[[Any], bool]):
        """Filters the population using a boolean function, keeping true values."""
        if col in self.cols:
            mask = np.full((self.data.shape[0],), True, dtype=bool)
            for row in range(self.data.shape[0]):
                mask[row] = fun(self.data[col][row])
            self.data = self.data[mask]
        else:
            raise Exception("No such column: {}".format(col))

    def unique(self, target_epoch=None, col="id"):
        """Reduces a population by eliminating duplicates with same oid.

        If target_epoch is not given, keep the latest instance found.
        If target_epoch is given, the last instance earlier than the epoch
        is kept, or the first after.
        If col is given, this is the field that will have only unique values
        """
        vmap = defaultdict(list)
        for ii, val in enumerate(self.data[col]):
            vmap[val].append(ii)

        # vmap will become catalogue of entries to delete,
        # so only pop()-ed itmes will remain
        for val in vmap:
            if len(vmap[val]) == 1:
                # Already unique, delete nothing
                vmap[val].pop(0)
                continue

            epochs = self.data["epoch"][vmap[val]]
            order = np.argsort(epochs)[::-1]  # vmap[val][order[0]] is latest
            if target_epoch is None:
                vmap[val].pop(order[0])
                continue
            if np.all(epochs > target_epoch):  # No earlier, pick earliest
                vmap[val].pop(order[-1])
                continue
            ii = np.argmax(epochs[order] < target_epoch)  # Find latest epoch < target
            vmap[val].pop(order[ii])

        # Must delete all items in one swoop, or indices will change under our feet
        deletions = reduce(lambda a, b: a + b, vmap.values())
        self.delete(deletions)

    @property
    def cols(self):
        """The columns"""
        return list(self.dtypes.keys())

    @property
    def keys(self):
        """The columns"""
        return list(self.dtypes.keys())

    @property
    def shape(self):
        """This is the shape of the internal data matrix"""
        shape = (len(self.data), len(self.keys))
        return shape

    def allocate(self, length: int):
        """Allocate the internal data array for assignment of objects.

        **Warning:** This removes all internal data.
        """
        _dtype = []
        for name, dt in self.dtypes.items():
            _dtype.append((name, dt))
        self.data = np.empty((length,), dtype=_dtype)

    def get_states(
        self,
        row_indecies: IndexLike | None = None,
        dtype: npt.DTypeLike | None = None,
    ):
        """Use the defined state parameters to get a copy of the states"""
        return self.get_fields(fields=self.state_fields, row_indecies=row_indecies, dtype=dtype)

    def get_fields(
        self,
        fields: list[str],
        row_indecies: IndexLike | None = None,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.NDArray:
        """Get the orbital elements for one row from internal data array.
        If the `dtype` is not `None`, a structured numpy array is returned,
        otherwise all data from the fields is typecast to the given `dtype`.
        """
        if row_indecies is None:
            row_indecies = slice(None, None, None)  # all

        states = self.data[row_indecies][fields]
        if dtype is not None:
            states_ = np.empty((len(states), len(fields)), dtype=dtype)

            for ind, key in enumerate(fields):
                states_[:, ind] = states[key].astype(dtype)
            states = states_

        return states

    def get_orbit(
        self,
        row_indecies: IndexLike | None = None,
        M_cent: float = pyorb.M_earth,
    ) -> pyorb.Orbit:
        """Get the one row from the population as a :class:`pyorb.Orbit` instance."""
        if row_indecies is None:
            row_indecies = slice(None, None, None)  # all

        fields = self.state_fields
        kwargs = {}
        if isinstance(row_indecies, int) or isinstance(row_indecies, np.integer):
            size = 1
        else:
            size = len(np.arange(len(self.data))[row_indecies])

        for key in fields:
            kwargs[key] = self.data[row_indecies][key]

        obj = pyorb.Orbit(
            M0=M_cent,
            degrees=self.degrees,
            type=self.anomly_type,
            auto_update=True,
            direct_update=True,
            num=size,
            **kwargs,
        )
        return obj

    def get_object(
        self,
        index: int,
        M_cent: float = pyorb.M_earth,
    ) -> SpaceObject:
        """Get the one row from the population as a `SpaceObject` instance."""

        orb = self.get_orbit(index, M_cent=M_cent)
        obj = SpaceObject(
            state=orb,
            frame=self.frame,
            epoch=Time(self.data["epoch"][index], format=self.epoch_format, scale=self.epoch_scale),
            properties={key: self.data[key][index] for key in self.property_fields},
            object_id=self.data["id"][index],
        )
        return obj

    def print(
        self,
        row_indecies: IndexLike | None = None,
        fields: list[str] | None = None,
    ) -> str:
        if row_indecies is None:
            row_indecies = slice(None, None, None)
        if fields is None:
            fields = self.keys

        data = self.data[row_indecies][fields]

        if isinstance(data, np.void):
            data = [[x for x in data]]

        return tabulate(data, headers=fields)

    def __str__(self):
        return self.print()

    def __iter__(self):
        self.__num = 0
        return self

    def __next__(self):
        if self.__num < self.data.shape[0]:
            ret = self.get_object(self.__num)
            self.__num += 1
            return ret
        else:
            raise StopIteration

    @property
    def generator(self):
        for obj in self:
            yield obj

    def save(self, fname: Path | str):
        if isinstance(fname, str):
            fname = Path(fname)

        with h5py.File(fname, "w") as hf:
            hf.create_dataset("data", data=self.data)
            hf.attrs["state_fields"] = self.state_fields
            hf.attrs["property_fields"] = self.property_fields
            hf.attrs["state_format"] = self.state_format
            hf.attrs["epoch_format"] = self.epoch_format
            hf.attrs["epoch_scale"] = self.epoch_scale
            hf.attrs["anomly_type"] = self.anomly_type
            hf.attrs["degrees"] = self.degrees
            hf.attrs["frame"] = self.frame

    @classmethod
    def load(cls, fname: Path | str) -> Self:
        if isinstance(fname, str):
            fname = Path(fname)

        with h5py.File(fname, "r") as hf:
            state_fields = copy.deepcopy(hf.attrs["state_fields"].tolist())
            data = hf["data"][()]
            dtypes = {key: data.dtype[key] for key in data.dtype.names}
            pop = cls(
                states=np.stack([data[key] for key in state_fields]),
                epochs=Time(
                    data["epoch"],
                    format=hf.attrs["epoch_format"],
                    scale=hf.attrs["epoch_scale"],
                ),
                frame=hf.attrs["frame"],
                parameters={
                    key: data[key] for key in copy.deepcopy(hf.attrs["property_fields"].tolist())
                },
                object_ids=data["id"],
                state_format=hf.attrs["state_format"],
                anomly_type=hf.attrs["anomly_type"],
                dtypes=dtypes,
                epoch_format=hf.attrs["epoch_format"],
                epoch_scale=hf.attrs["epoch_scale"],
                degrees=hf.attrs["degrees"],
            )

        return pop
