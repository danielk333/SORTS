#!/usr/bin/env python

'''Defines a population of space objects in the form of a class.

'''

#Python standard import
import copy
import pathlib
from collections import defaultdict
from functools import reduce

#Third party import
import h5py
import numpy as np
import pyorb
from tabulate import tabulate
from astropy.time import Time

#Local import
from .. import space_object as so
from .. import plotting
from .. import constants

class Population:
    '''Encapsulates a population of space objects as an array and functions for returning instances of space objects.

    **Default columns:**

        * 0: oid - Object ID
        * 1: a - Semi-major axis in m
        * 2: e - Eccentricity 
        * 3: i - Inclination in degrees
        * 4: raan - Right Ascension of ascending node in degrees
        * 5: aop - Argument of perihelion in degrees
        * 6: mu0 - Mean anoamly in degrees
        * 7: mjd0 - Epoch of object given in Modified Julian Days

    Any column that is added will have its name used in initializing the Space object.

    A population's column data can also be accessed as a python dictionary or a table according to row number, e.g.

    .. code-block:: python

        #this returns all Right Ascension of ascending node as a numpy vector
        vector = my_population['raan']
        
        #this gets row number 3 (since we use 0 indexing)
        row = my_population[2]

    but it is also configured to be able to try to convert to uniform type array and perform numpy like slices. If a column data type cannot be converted to the default data type numpy.nan is inserted instead.

    .. code-block:: python

        #This will convert the internal data structure to a uniform type array and select all rows and columns 4 and onwards. This is time-consuming on large populations as it actually copies to data.
        vector = my_population[:,4:]

        # This is significantly faster as a single column is easy to extract and no conversion is needed
        data_point = my_population[123,2]

    This indexing system can also be used for data manipulation:

    .. code-block:: python

        my_population['raan'] = vector
        my_population[2] = row
        my_population[:,4:] = matrix
        my_population[123,2] = 2.3
        my_population[123,11] = 'test'


    Notice that in the above example the value to be assigned always has the correct size corresponding to the index and slices, a statement like :code:`x[:,3:7] = 3` is not possible, instead one would write :code:`x[:,3:7] = np.full((len(pop), 4), 3.0, dtype='f')`.


    #TODO: THESE HAVE CHANGED, UPDATE THEM!!!!

    :ivar numpy.ndarray objs: Array containing population data. Rows correspond to objects and columns to variables.
    :ivar str name: Name of population.
    :ivar list fields: List of strings containing column descriptions.
    :ivar list space_object_uses: List of booleans describing what columns should be included when initializing a space object. This allows for extra data to be stored in the population without passing it to the space object.
    :ivar PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :ivar dict propagator_options: Propagator initialization keyword arguments.
    
    :param str name: Name of population.
    :param list fields: List of strings containing column descriptions for addition data besides the default columns.
    :param list dtypes: List of strings containing numpy data type description. Defaults to 'f'.
    :param list space_object_fields: List of booleans describing what columns should be included when initializing a space object. This allows for extra data to be stored in the population without passing it to the space object.
    :param PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :param dict propagator_options: Propagator initialization keyword arguments.
    
    '''

    _default_dtype = 'float64'
    _default_fields = ['oid','a','e','i','raan','aop','mu0','mjd0']
    _default_state_fields = ['a','e','i','raan','aop','mu0']
    _default_epoch = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'}

    def __init__(self,
                fields = None,
                dtypes = None,
                space_object_fields = None,
                state_fields = None,
                epoch_field = None,
                propagator = None,
                propagator_args = {},
                propagator_options = {},
            ):
        
        if fields is None:
            fields = copy.copy(Population._default_fields)
        self.fields = fields

        self.space_object_fields = space_object_fields
        if dtypes is None:
            self.dtypes = [Population._default_dtype]*len(self.fields)
        else:
            self.dtypes = dtypes

        for dt in self.dtypes:
            if np.dtype(dt).char == 'U':
                raise TypeError('Initialized Population cannot use the save function with Unicode [U] numpy strings, try using ASCII [S] strings instead.')

        if state_fields is None:
            state_fields = []
            for key in Population._default_state_fields:
                if key in self.fields:
                    state_fields.append(key)
        self.state_fields = state_fields

        if epoch_field is None:
            epoch_field = copy.deepcopy(Population._default_epoch)
        else:
            if isinstance(epoch_field, str):
                epoch_field_ = epoch_field
                epoch_field = copy.deepcopy(Population._default_epoch)
                epoch_field['field'] = epoch_field_

        self.epoch_field = epoch_field

        self.allocate(0)
        
        self.propagator = propagator
        self.propagator_options = propagator_options
        self.propagator_args = propagator_args

    def __len__(self):
        return(self.data.shape[0])


    @property
    def out_frame(self):
        if 'settings' not in self.propagator_options:
            return None
        if 'out_frame' not in self.propagator_options['settings']:
            return None 

        return self.propagator_options['settings']['out_frame']
            

    @out_frame.setter
    def out_frame(self, val):
        if 'settings' not in self.propagator_options:
            self.propagator_options['settings'] = {}
        self.propagator_options['settings']['out_frame'] = val


    @property
    def in_frame(self):
        if 'settings' not in self.propagator_options:
            return None
        if 'in_frame' not in self.propagator_options['settings']:
            return None 

        return self.propagator_options['settings']['in_frame']
            

    @in_frame.setter
    def in_frame(self, val):
        if 'settings' not in self.propagator_options:
            self.propagator_options['settings'] = {}
        self.propagator_options['settings']['in_frame'] = val



    def copy(self):
        '''Return a copy of the current Population instance.
        '''
        pop = Population(
                fields = copy.deepcopy(self.fields),
                dtypes = copy.deepcopy(self.dtypes),
                space_object_fields = copy.deepcopy(self.space_object_fields),
                state_fields = copy.deepcopy(self.state_fields),
                epoch_field = copy.deepcopy(self.epoch_field),
                propagator = self.propagator,
                propagator_args = copy.deepcopy(self.propagator_args),
                propagator_options = copy.deepcopy(self.propagator_options),
            )

        pop.data = self.data.copy()
        return pop

    def delete(self, inds):
        '''Remove the rows according to the given indices. Supports single index, iterable of indices and slices.
        '''
        if isinstance(inds, int):
            inds = [inds]
        elif isinstance(inds, slice):
            _inds = range(self.data.shape[0])
            inds = _inds[inds]
        elif not (isinstance(inds, list) or isinstance(inds, np.ndarray)):
            raise Exception('Cannot delete indecies given with type {}'.format(type(inds)))

        mask = np.full( (self.data.shape[0],), True, dtype=np.bool)
        for ind in inds:
            mask[ind] = False
        self.data=self.data[ mask ]


    def filter(self,col,fun):
        '''Filters the population using a boolean function, keeping true values.

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

        '''
        if col in self.fields:
            ind = self.fields.index(col)
            mask = np.full( (self.data.shape[0],), True, dtype=np.bool)
            for row in range(self.data.shape[0]):
                mask[row] = fun(self.data[col][row])
            self.data=self.data[ mask ]
        else:
            raise Exception('No such column: {}'.format(col))


    def unique(self, target_epoch=None, col='oid'):
        '''Reduces a population by eliminating duplicates with same oid.

        If target_epoch is not given, keep the latest instance found.
        If target_epoch is given, the last instance earlier than the epoch
        is kept, or the first after.
        If col is given, this is the field that will have only unique values
        '''
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

            epochs = self.data['mjd0'][vmap[val]]
            order = np.argsort(epochs)[::-1]        # vmap[val][order[0]] is latest
            if target_epoch is None:
                vmap[val].pop(order[0])
                continue
            if np.all(epochs > target_epoch):           # No earlier, pick earliest
                vmap[val].pop(order[-1])
                continue
            ii = np.argmax(epochs[order] < target_epoch)        # Find latest epoch < target
            vmap[val].pop(order[ii])

        # Must delete all items in one swoop, or indices will change under our feet
        deletions = reduce(lambda a, b: a+b, vmap.values())
        self.delete(deletions)


    @property
    def shape(self):
        '''This is the shape of the internal data matrix
        '''
        shape = (self.data.shape[0],len(self.fields))
        return shape


    def allocate(self, length):
        '''Allocate the internal data array for assignment of objects.

        **Warning:** This removes all internal data.

        **Example:**

        Create a population with two objects. Here the :code:`load_data` function is a fictional function that creates a row with the needed data.
    
        .. code-block:: python

            from population import Population
            from my_data_loader import load_data

            my_pop = Population(
                name='two objects',
                extra_columns = ['m', 'color'],
                dtypes = ['Float64', 'U20'],
                space_object_uses = [True, False],
            )
            
            print(len(my_pop)) #will output 0
            my_pop.allocate(2)
            print(len(my_pop)) #will output 2

            my_pop.data[0] = load_data('obj1')
            my_pop.data[1] = load_data('obj2')

        '''
        _dtype = []
        for name, dt, ind in zip(self.fields, self.dtypes, range(len(self.fields))):
            _dtype.append( (name, dt))

        self.data = np.empty((length,), dtype=_dtype)


    def get_states(self, n=None, named=True, dtype=None):
        '''Use the defined state parameters to get a copy of the states
        '''
        return self.get_fields(fields = self.state_fields, n=n, named=named, dtype=dtype)


    def get_fields(self, fields, n=None, named=True, dtype=None):
        '''Get the orbital elements for one row from internal data array.

        :param int/slice/list n: Row number(s).
        :param list fields: List of fields to get data for
        :param bool named: return a named numpy array or a unnamed one. If True, all dtypes are cast as the first fields.
        '''
        if n is None:
            n = slice(None,None,None) #all

        states = self.data[n][fields]
        if not named:
            if dtype is None:
                dtype = states.dtype[0]
            states_ = np.empty((len(states), len(fields)), dtype=dtype)
            for ind, key in enumerate(states.dtype.names):
                states_[:,ind] = states[key].astype(dtype)
            states = states_
            del states_

        return states


    def get_orbit(self, n, fields=None, M_cent=pyorb.M_earth, degrees=True, anomaly='mean'):
        '''Get the one row from the population as a :class:`pyorb.Orbit` instance.
        '''

        if fields is None:
            fields = self.state_fields

        kwargs = {}

        for key in fields:
            kwargs[key] = self.data[n][key]

        #TODO: generalize this better
        if 'aop' in kwargs:
            kwargs['omega'] = kwargs.pop('aop')
        if 'raan' in kwargs:
            kwargs['Omega'] = kwargs.pop('raan')
        if 'mu0' in kwargs:
            kwargs['anom'] = kwargs.pop('mu0')


        for key in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
            if key in kwargs:
                kwargs[key.lower()] = kwargs.pop(key)

        obj=pyorb.Orbit(
            M0 = M_cent, 
            degrees = degrees,
            type=anomaly,
            auto_update = True, 
            direct_update = True,
            num = 1,
            **kwargs
        )
        return obj



    def get_object(self, n):
        '''Get the one row from the population as a :class:`space_object.SpaceObject` instance.
        '''
        parameters = {}
        if self.space_object_fields is not None:
            for key in self.space_object_fields:
                parameters[key] = self.data[key][n]

        cart_state = True
        kep_state = True
        for key in pyorb.Orbit.CARTESIAN:
            if key not in self.state_fields:
                cart_state = False
        for key in ['a', 'e', 'i']:
            if key not in self.state_fields:
                kep_state = False
        if 'omega' not in self.state_fields and 'aop' not in self.state_fields:
            kep_state = False
        if 'Omega' not in self.state_fields and 'raan' not in self.state_fields:
            kep_state = False
        if 'anom' not in self.state_fields and 'mu0' not in self.state_fields:
            kep_state = False

        kwargs = {}
        if kep_state or cart_state:
            for key in self.state_fields:
                kwargs[key] = self.data[n][key]
        else:
            kwargs['state'] = self.data[n][self.state_fields]

        if 'oid' in self.fields:
            kwargs['oid'] = self.data[n]['oid']

        obj=so.SpaceObject(
            propagator = self.propagator,
            propagator_options = self.propagator_options,
            propagator_args = self.propagator_args,
            parameters = parameters,
            epoch=Time(
                self.data[self.epoch_field['field']][n], 
                format=self.epoch_field['format'], 
                scale=self.epoch_field['scale'],
            ),
            **kwargs
        )
        return obj


    def add_field(self, name, dtype=None):
        '''Add a field to the population data.
        '''
        data_tmp = self.data.copy()
        self.fields.append(name)
        if dtype is None:
            dtype = Population._default_dtype
        self.dtypes.append(dtype)

        self.allocate(len(data_tmp))
        for key in self.fields:
            if key != name:
                self.data[key] = data_tmp[key]


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


    def __getitem__(self, key):

        if isinstance(key, tuple):
            if len(key) == 2:

                is_slice = False
                for _key in key:
                    if isinstance(_key, slice):
                        is_slice = True

                if is_slice:
                    tmp_data = np.empty((self.data.shape[0], len(self.fields)), dtype=np.dtype(self._default_dtype))
                    for ind, col in enumerate(self.fields):
                        col_data = self.data[col]
                        try:
                            tmp_data[0,ind] = col_data[0]
                            convertable = True
                        except ValueError:
                            convertable = False

                        if convertable:
                            tmp_data[:,ind] = col_data
                        else:
                            tmp_data[:,ind] = np.nan
                    return tmp_data[key[0],key[1]]
                else:
                    return self.data[self.fields[key[1]]][key[0]]
            else:
                raise Exception('Too many incidences given, only supports 2')
        elif isinstance(key, str):
            if key in self.fields:
                return self.data[key]
            else:
                raise Exception('No such column: {}'.format(key))
        elif isinstance(key, int):
            if key < self.data.shape[0]:
                return self.data[key]
            else:
                raise Exception('Row number {} outside range of population {}'.format(key, self.data.shape[0]))
        else:
            raise Exception('Key type "{}" not supported'.format(type(key)))


    def __setitem__(self, key, data):

        if isinstance(key, tuple):
            if len(key) == 2:
                row_iter = list(range(len(self)))[key[0]]
                head_iter = self.fields[key[1]]

                if not isinstance(row_iter,list):
                    row_point = True
                    row_iter = [row_iter]
                else:
                    row_point = False
                if not isinstance(head_iter,list):
                    col_point = True
                    head_iter = [head_iter]
                else:
                    col_point = False

                if row_point and col_point:
                    self.data[head_iter[0]][row_iter[0]] = data
                else:
                    for coli, head in enumerate(head_iter):
                        for rowi, row_num in enumerate(row_iter):
                            if col_point:
                                self.data[head][row_num] = data[rowi]
                            elif row_point:
                                self.data[head][row_num] = data[coli]
                            else:
                                self.data[head][row_num] = data[rowi][coli]
            else:
                raise Exception('Too many incidences given, only supports 2')
        elif isinstance(key, str):
            if key in self.fields:
                self.data[key] = data
            else:
                raise Exception('No such column: {}'.format(key))
        elif isinstance(key, int):
            if key < self.data.shape[0]:
                for coli, head in enumerate(self.fields):
                    self.data[head][key] = data[coli]
            else:
                raise Exception('Row number {} outside range of population {}'.format(key, self.data.shape[0]))
        else:
            raise Exception('Key type "{}" not supported'.format(type(key)))

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
        if isinstance(fname, str):
            fname = pathlib.Path(fname)


        with h5py.File(fname,"w") as hf:
            hf.create_dataset('data', data=self.data)
            hf.attrs['fields'] = self.fields
            hf.attrs['space_object_fields'] = self.space_object_fields
            hf.attrs['dtypes'] = self.dtypes
            hf.attrs['epoch_field'] = [x for x in self.epoch_field.items()]
            hf.attrs['state_fields'] = self.state_fields


    @classmethod
    def load(cls, fname, propagator, propagator_options = {}, propagator_args = {}):
        if isinstance(fname, str):
            fname = pathlib.Path(fname)

        with h5py.File(fname,"r") as hf:
            pop = cls(
                fields = copy.deepcopy(hf.attrs['fields'].tolist()),
                dtypes = copy.deepcopy(hf.attrs['dtypes'].tolist()),
                space_object_fields = copy.deepcopy(hf.attrs['space_object_fields'].tolist()),
                state_fields = copy.deepcopy(hf.attrs['state_fields'].tolist()),
                epoch_field = {key:val for key, val in hf.attrs['epoch_field']},
                propagator = propagator,
                propagator_options = propagator_options,
                propagator_args = propagator_args,
            )
        
            pop.data = hf['data'][()]

        return pop



#python 2.7 compliance
Population.next = Population.__next__

