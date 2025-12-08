#!/usr/bin/env python

"""Defines a population of space objects in the form of a class."""
import copy
import pathlib
from collections import defaultdict, OrderedDict
from functools import reduce

import h5py
import numpy as np
import numpy.typing as npt
import pyorb
from tabulate import tabulate
from astropy.time import Time

from sorts.types import StateType, NDArray_6xN, NDArray_N
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
        parameters: dict[str, NDArray_N],
        object_ids: NDArray_N | None,
        state_format: StateType = "kepler",
        dtypes: dict[str, npt.DTypeLike] | None = None,
        default_dtype: npt.DTypeLike = np.float64,
        epoch_format: str = "mjd",
        epoch_scale: str = "utc",
    ):
        assert states.shape[1] == len(epochs)
        match state_format:
            case "kepler":
                state_keys = pyorb.Orbit.KEPLER
            case "cartesian":
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
        self.state_format = state_format
        self.epoch_format = epoch_format
        self.epoch_scale = epoch_scale

        data_keys = ["id", "epoch"] + state_keys + list(parameters.keys())
        self.dtypes = OrderedDict()
        for key in data_keys:
            dt = dtypes[key] if key in dtypes else default_dtype
            self.dtypes[key] = dt

        self.allocate(len(epochs))
        self.data["id"] = np.arange(len(epochs)) if object_ids is None else object_ids
        self.data["epoch"] = getattr(epochs, epoch_format).values
        for key, ind in enumerate(state_keys):
            self.data[key] = states[ind, :]
        for key, vals in parameters.items():
            self.data[key] = vals

    def __len__(self) -> int:
        return len(self.data)

    def copy(self):
        """Return a copy of the current Population instance."""
        pop = Population()
        raise NotImplementedError()
        pop.data = self.data.copy()
        return pop

    def delete(self, inds):
        """Remove the rows according to the given indices.
        Supports single index, iterable of indices and slices.
        """
        if isinstance(inds, int):
            inds = [inds]
        elif isinstance(inds, slice):
            _inds = range(self.data.shape[0])
            inds = _inds[inds]
        elif not (isinstance(inds, list) or isinstance(inds, np.ndarray)):
            raise Exception("Cannot delete indecies given with type {}".format(type(inds)))

        mask = np.full((self.data.shape[0],), True, dtype=bool)
        for ind in inds:
            mask[ind] = False
        self.data = self.data[mask]

    def filter(self, col, fun):
        """Filters the population using a boolean function, keeping true values.

        :param str col: Column to filter, must match exactly one entry in the :code:`header` attribute.
        :param function fun: Function that returns boolean array used for filtering.

        **Example:**

        Filter Master population keeping only objects below 45.0 degrees inclination.

        .. code-block:: python

            from population_library import master_catalog

            master = master_catalog()
            master.filter(
                col='i',
                fun=lambda inc: inc < 45.0,
            )

        """
        if col in self.fields:
            mask = np.full((self.data.shape[0],), True, dtype=bool)
            for row in range(self.data.shape[0]):
                mask[row] = fun(self.data[col][row])
            self.data = self.data[mask]
        else:
            raise Exception("No such column: {}".format(col))

    def unique(self, target_epoch=None, col="oid"):
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

            epochs = self.data[self.epoch_field["field"]][vmap[val]]
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
    def keys(self):
        """The  property."""
        return list(self.dtypes.keys())

    @property
    def shape(self):
        """This is the shape of the internal data matrix"""
        shape = (len(self.data), len(self.keys))
        return shape

    def allocate(self, length):
        """Allocate the internal data array for assignment of objects.

        **Warning:** This removes all internal data.
        """
        _dtype = []
        for name, dt in self.dtypes.items():
            _dtype.append((name, dt))
        self.data = np.empty((length,), dtype=_dtype)

    def get_states(self, n=None, named=True, dtype=None):
        """Use the defined state parameters to get a copy of the states"""
        return self.get_fields(fields=self.state_fields, n=n, named=named, dtype=dtype)

    def get_fields(self, fields, n=None, named=True, dtype=None):
        """Get the orbital elements for one row from internal data array.

        :param int/slice/list n: Row number(s).
        :param list fields: List of fields to get data for
        :param bool named: return a named numpy array or a unnamed one. If True, all dtypes are cast as the first fields.
        """
        if n is None:
            n = slice(None, None, None)  # all

        states = self.data[n][fields]
        if not named:
            if dtype is None:
                dtype = states.dtype[0]
            states_ = np.empty((len(states), len(fields)), dtype=dtype)
            for ind, key in enumerate(states.dtype.names):
                states_[:, ind] = states[key].astype(dtype)
            states = states_
            del states_

        return states

    def get_orbit(self, n, fields=None, M_cent=pyorb.M_earth, degrees=True, anomaly="mean"):
        """Get the one row from the population as a :class:`pyorb.Orbit` instance."""
        raise NotImplementedError()

        if fields is None:
            fields = self.state_fields

        kwargs = {}

        for key in fields:
            kwargs[key] = self.data[n][key]

        # TODO: generalize this better
        if "aop" in kwargs:
            kwargs["omega"] = kwargs.pop("aop")
        if "raan" in kwargs:
            kwargs["Omega"] = kwargs.pop("raan")
        if "mu0" in kwargs:
            kwargs["anom"] = kwargs.pop("mu0")

        for key in ["X", "Y", "Z", "VX", "VY", "VZ"]:
            if key in kwargs:
                kwargs[key.lower()] = kwargs.pop(key)

        obj = pyorb.Orbit(
            M0=M_cent,
            degrees=degrees,
            type=anomaly,
            auto_update=True,
            direct_update=True,
            num=1,
            **kwargs,
        )
        return obj

    def get_object(self, n):
        """Get the one row from the population as a :class:`space_object.SpaceObject` instance."""
        parameters = {}
        raise NotImplementedError()
        if self.space_object_fields is not None:
            for key in self.space_object_fields:
                parameters[key] = self.data[key][n]

        cart_state = True
        kep_state = True
        for key in pyorb.Orbit.CARTESIAN:
            if key not in self.state_fields:
                cart_state = False
        for key in ["a", "e", "i"]:
            if key not in self.state_fields:
                kep_state = False
        if "omega" not in self.state_fields and "aop" not in self.state_fields:
            kep_state = False
        if "Omega" not in self.state_fields and "raan" not in self.state_fields:
            kep_state = False
        if "anom" not in self.state_fields and "mu0" not in self.state_fields:
            kep_state = False

        kwargs = {}
        if kep_state or cart_state:
            for key in self.state_fields:
                kwargs[key] = self.data[n][key]
        else:
            kwargs["state"] = self.data[n][self.state_fields]

        if "oid" in self.fields:
            kwargs["oid"] = self.data[n]["oid"]

        obj = SpaceObject(
            propagator=self.propagator,
            propagator_options=self.propagator_options,
            propagator_args=self.propagator_args,
            parameters=parameters,
            epoch=Time(
                self.data[self.epoch_field["field"]][n],
                format=self.epoch_field["format"],
                scale=self.epoch_field["scale"],
            ),
            **kwargs,
        )
        return obj

    def print(self, n=None, fields=None):
        if n is None:
            n = slice(None, None, None)
        if fields is None:
            fields = self.fields

        data = self.data[n][fields]

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

    def save(self, fname):
        raise NotImplementedError()
        if isinstance(fname, str):
            fname = pathlib.Path(fname)

        with h5py.File(fname, "w") as hf:
            hf.create_dataset("data", data=self.data)
            hf.attrs["fields"] = self.fields
            hf.attrs["space_object_fields"] = self.space_object_fields
            hf.attrs["dtypes"] = self.dtypes
            hf.attrs["epoch_field"] = [x for x in self.epoch_field.items()]
            hf.attrs["state_fields"] = self.state_fields

    @classmethod
    def load(cls, fname):
        raise NotImplementedError()
        if isinstance(fname, str):
            fname = pathlib.Path(fname)

        with h5py.File(fname, "r") as hf:
            pop = cls(
                fields=copy.deepcopy(hf.attrs["fields"].tolist()),
                dtypes=copy.deepcopy(hf.attrs["dtypes"].tolist()),
                space_object_fields=copy.deepcopy(hf.attrs["space_object_fields"].tolist()),
                state_fields=copy.deepcopy(hf.attrs["state_fields"].tolist()),
                epoch_field={key: val for key, val in hf.attrs["epoch_field"]},
            )

            pop.data = hf["data"][()]

        return pop
