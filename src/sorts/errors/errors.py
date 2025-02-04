#!/usr/bin/env python

"""Errors

"""
# Python standard import
from abc import ABC, abstractmethod
import types
import pathlib

# Third party import
import numpy as np


# Local import


class Errors(ABC):
    """A standardized way for adding random errors to data

    The methods corresponding to the variables should take the data and return
    """

    VARIABLES = []

    def __init__(self, seed=None):
        self.seed = seed
        self._check_methods()

    def set_numpy_seed(self):
        """This should be called before any error generating methods if reproduction of results needs to be ensured."""
        if self.seed is not None:
            self.seed += 1
            np.random.seed(self.seed)

    def get(self, var, data, *args, **kwargs):
        func = getattr(self, var)
        return func(data, *args, **kwargs)

    def _check_method(self, name):
        return hasattr(self, name) and type(getattr(self, name)) == types.MethodType

    def _check_methods(self):
        for var in self.VARIABLES:
            if not self._check_method(var):
                raise AttributeError(
                    f"Could not find implemented error generating method for {var}"
                )
