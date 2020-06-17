#!/usr/bin/env python

'''Defines a population of space objects in the form of a class.

'''

#Python standard import
import copy

#Third party import
import h5py
import numpy as np
import pyorb

#Local import
from . import space_object as so
from . import plotting
from . import constants

class Population:
    '''Encapsulates a population of space objects as an array and functions for returning instances of space objects.

    **Default columns:**

        * 0: oid - Object ID
        * 1: a - Semi-major axis in km
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


    :ivar numpy.ndarray objs: Array containing population data. Rows correspond to objects and columns to variables.
    :ivar str name: Name of population.
    :ivar list header: List of strings containing column descriptions.
    :ivar list space_object_uses: List of booleans describing what columns should be included when initializing a space object. This allows for extra data to be stored in the population without passing it to the space object.
    :ivar PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :ivar dict propagator_options: Propagator initialization keyword arguments.
    
    :param str name: Name of population.
    :param list extra_columns: List of strings containing column descriptions for addition data besides the default columns.
    :param list dtypes: List of strings containing numpy data type description. Defaults to 'f'.
    :param list space_object_uses: List of booleans describing what columns should be included when initializing a space object. This allows for extra data to be stored in the population without passing it to the space object.
    :param PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :param dict propagator_options: Propagator initialization keyword arguments.
    
    '''

    _default_dtype = 'float64'

    def __init__(self,
                name='Unnamed population',
                extra_columns = [],
                dtypes = [],
                space_object_uses = [],
                propagator = None,
                propagator_args = {},
                propagator_options = {},
            ):
        self.name = name
        
        self.header = ['oid','a','e','i','raan','aop','mu0','mjd0']
        self.space_object_uses = [True]*len(self.header)
        self.dtypes = [self._default_dtype]*len(self.header)

        self.header += extra_columns
        self.space_object_uses += space_object_uses

        if len(dtypes) == 0:
            self.dtypes += [self._default_dtype]*len(extra_columns)
        elif len(dtypes) != len(extra_columns):
            raise Exception('Not enough dtypes given for extra columns')
        else:
            self.dtypes += dtypes
        
        self.allocate(0)
        
        self.propagator = propagator
        self.propagator_options = propagator_options

    def __len__(self):
        return(self.objs.shape[0])

    def copy(self):
        '''Return a copy of the current Population instance.
        '''
        pop = Population(
                propagator = self.propagator,
                name=self.name,
                extra_columns = [],
                dtypes = [],
                space_object_uses = [],
                propagator_args = self.propagator_args,
                propagator_options = copy.deepcopy(self.propagator_options),
            )

        pop.header = copy.deepcopy(self.header)
        pop.space_object_uses = copy.deepcopy(self.space_object_uses)
        pop.dtypes = copy.deepcopy(self.dtypes)
        pop.objs = self.objs.copy()
        return pop

    def delete(self, inds):
        '''Remove the rows according to the given indices. Supports single index, iterable of indices and slices.
        '''
        if isinstance(inds, int):
            inds = [inds]
        elif isinstance(inds, slice):
            _inds = range(self.objs.shape[0])
            inds = _inds[inds]
        elif not (isinstance(inds, list) or isinstance(inds, np.ndarray)):
            raise Exception('Cannot delete indecies given with type {}'.format(type(inds)))

        mask = np.full( (self.objs.shape[0],), True, dtype=np.bool)
        for ind in inds:
            mask[ind] = False
        self.objs=self.objs[ mask ]


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
        if col in self.header:
            ind = self.header.index(col)
            mask = np.full( (self.objs.shape[0],), True, dtype=np.bool)
            for row in range(self.objs.shape[0]):
                mask[row] = fun(self.objs[col][row])
            self.objs=self.objs[ mask ]
        else:
            raise Exception('No such column: {}'.format(col))

    def _str_header(self):
        ret_str = ''
        _sep = '  |  '
        header = ['{:<12}']*len(self.header)
        header = [header[ind].format(nm) for ind, nm in enumerate(self.header)]
        header = _sep.join(header)

        ret_str += header + '\n'
        ret_str += '-'*len(header)
        return ret_str

    def _str_row(self,n):
        ret_str = ''
        row = self.objs[n]
        _sep = '  |  '
        _row = [None]*len(self.header)
        for ind, field in enumerate(self.header):
            if np.issubdtype(self.objs.dtype[field], np.inexact):
                _row[ind] = '{:<12.4f}'.format(row[field])
            else:
                _row[ind] = '{!r:<12}'.format(row[field])

        return _sep.join(_row)

    @property
    def shape(self):
        '''This is the shape of the internal data matrix
        '''
        shape = (self.objs.shape[0],len(self.header))
        return shape


    def print_row(self,n):
        '''Print a specific row with Header information.
        '''
        head = self._str_header()
        row = self._str_row(n)
        print(head)
        print(row)

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

            my_pop.objs[0] = load_data('obj1')
            my_pop.objs[1] = load_data('obj2')

        '''
        _dtype = []
        for name, dt, ind in zip(self.header, self.dtypes, range(len(self.header))):
            _dtype.append( (name, dt))

        self.objs = np.empty((length,), dtype=_dtype)

    def get_states(self, M_cent = constants.M_earth):
        '''Use the orbital parameters and get the state.'''
        orbs = self.get_all_orbits(order_angs = True).T
        orbs[:,5] = pyorb.mean_to_true(orbs[:,5], orbs[:,1], radians=False)
        if 'm' in self.header:
            mv = self.objs['m']
        else:
            mv = np.zeros(len(self), dtype=self._default_dtype)
        states = pyorb.kep_to_cart(orbs, m=mv, M_cent=M_cent, radians=False).T
        return states



    def get_orbit(self,n, order_angs=False):
        '''Get the orbital elements for one row from internal data array.

        :param int n: Row number.
        :param bool order_angs: Order the orbital element angles according to aop before raan or not.
        '''
        if order_angs:
            dat = ['a','e','i','aop','raan','mu0']
        else:
            dat = ['a','e','i','raan','aop','mu0']

        row = np.empty((1, 6), dtype=np.dtype(self._default_dtype))
        for coln, col in enumerate(dat):
            row[coln] = self.objs[col][n]
        return row

    def get_all_orbits(self, order_angs=False):
        '''Get the orbital elements for all rows from internal data array.

        :param bool order_angs: Order the orbital element angles according to aop before raan or not.
        '''
        if order_angs:
            dat = ['a','e','i','aop','raan','mu0']
        else:
            dat = ['a','e','i','raan','aop','mu0']

        ret = np.empty((self.objs.shape[0], 6), dtype=np.dtype(self._default_dtype))
        for ind in range(self.objs.shape[0]):
            for coln, col in enumerate(dat):
                ret[ind,coln] = self.objs[col][ind]
        return ret

    def get_object(self,n):
        '''Get the one row from the population as a :class:`space_object.SpaceObject` instance.
        '''
        kw = {}
        for head, use in zip(self.header, self.space_object_uses):
            if use:
                kw[head] = self.objs[head][n]

        o=so.SpaceObject(
            propagator = self.propagator,
            propagator_options = self.propagator_options,
            **kw
        )
        return o

    def object_generator(self):
        '''Return a generator that iterates trough the entire population returning space objects.
        '''
        for ind in range(self.objs.shape[0]):
            yield self.get_object(ind)


    def add_column(self, name, dtype=_default_dtype, space_object_uses=False):
        '''Add a column to the population data.
        '''
        data_tmp = self.objs.copy()
        self.header.append(name)
        self.dtypes.append(dtype)
        self.space_object_uses.append(space_object_uses)
        self.allocate(len(data_tmp))
        for ind in range(self.objs.shape[0]):
            for head in self.header:
                if name != head:
                    self.objs[ind][head] = data_tmp[ind][head]


    def __str__(self):
        ret_str = self._str_header() + '\n'
        for ind in range(len(self)):
            ret_str += self._str_row(ind)
            ret_str += '\n'
        return ret_str

    def __getitem__(self, key):

        if isinstance(key, tuple):
            if len(key) == 2:

                is_slice = False
                for _key in key:
                    if isinstance(_key, slice):
                        is_slice = True

                if is_slice:
                    tmp_data = np.empty((self.objs.shape[0], len(self.header)), dtype=np.dtype(self._default_dtype))
                    for ind, col in enumerate(self.header):
                        col_data = self.objs[col]
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
                    return self.objs[self.header[key[1]]][key[0]]
            else:
                raise Exception('Too many incidences given, only supports 2')
        elif isinstance(key, str):
            if key in self.header:
                return self.objs[key]
            else:
                raise Exception('No such column: {}'.format(key))
        elif isinstance(key, int):
            if key < self.objs.shape[0]:
                return self.objs[key]
            else:
                raise Exception('Row number {} outside range of population {}'.format(key, self.objs.shape[0]))
        else:
            raise Exception('Key type "{}" not supported'.format(type(key)))


    def __setitem__(self, key, data):

        if isinstance(key, tuple):
            if len(key) == 2:
                row_iter = list(range(len(self)))[key[0]]
                head_iter = self.header[key[1]]

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
                    self.objs[head_iter[0]][row_iter[0]] = data
                else:
                    for coli, head in enumerate(head_iter):
                        for rowi, row_num in enumerate(row_iter):
                            if col_point:
                                self.objs[head][row_num] = data[rowi]
                            elif row_point:
                                self.objs[head][row_num] = data[coli]
                            else:
                                self.objs[head][row_num] = data[rowi][coli]
            else:
                raise Exception('Too many incidences given, only supports 2')
        elif isinstance(key, str):
            if key in self.header:
                self.objs[key] = data
            else:
                raise Exception('No such column: {}'.format(key))
        elif isinstance(key, int):
            if key < self.objs.shape[0]:
                for coli, head in enumerate(self.header):
                    self.objs[head][key] = data[coli]
            else:
                raise Exception('Row number {} outside range of population {}'.format(key, self.objs.shape[0]))
        else:
            raise Exception('Key type "{}" not supported'.format(type(key)))

    def __iter__(self):
        self.__num = 0
        return self


    def __next__(self):
        if self.__num < self.objs.shape[0]:
            ret = self.objs[self.__num]
            self.__num += 1
            return ret
        else:
            raise StopIteration

    def save(self, fname):
        with h5py.File(fname,"w") as hf:
            hf.create_dataset('objs', data=self.objs)
            hf.create_dataset('header',
                data=np.array(self.header),
            )
            hf.create_dataset('space_object_uses',
                data=np.array(self.space_object_uses),
            )
            hf.create_dataset('dtypes',
                data=np.array(self.dtypes),
            )
            hf.attrs['name'] = self.name

    @classmethod
    def load(cls, fname, propagator, propagator_options = {}, propagator_args = {}):
        pop = cls(
            propagator = propagator,
            name='Unnamed population',
            extra_columns = [],
            dtypes = [],
            space_object_uses = [],
            propagator_args = propagator_args,
            propagator_options = propagator_options,
        )
        with h5py.File(fname,"r") as hf:
            pop.objs = hf['objs'].value

            pop.header = hf['header'].value.tolist()
            pop.space_object_uses = hf['space_object_uses'].value.tolist()
            pop.dtypes = hf['dtypes'].value.tolist()

            pop.name = hf.attrs['name']

        return pop



    def plot_distribution(self, dist, label = None, logx = False, logy = False, log_freq = False):
        '''Plot the distribution of parameter(s) or all orbits of this population.

        :param str/list dist: Name of parameter as given by :code:`population.header` or :code:`'orbits'` to plot all orbits. If a list, length of list must be exactly 2 and will produce a 2d distribution instead.
        :param str/list label: Used if parameter(s) distribution is plotted to label the axis.
        :param bool logx: Determines if x-axis is logarithmic or not.
        :param bool logy: Determines if y-axis is logarithmic or not.
        :param bool log_freq: Determines if frequency is logarithmic or not.
        '''
        if isinstance(dist, str):        
            if dist=='orbits':
                fig, ax = plotting.orbits(
                    self.get_all_orbits(order_angs=True), 
                    title = "Orbit distribution: {}".format(self.name),
                    show = False,
                )
            elif dist in self.header:
                if label is None:
                    x_label = dist
                else:
                    x_label = label
                fig, ax = plotting.hist(
                    self.objs[dist],
                    title = "{} distribution: {}".format(dist, self.name),
                    show = False,
                    xlabel = x_label,
                    logx = logx,
                    logy = log_freq,
                )
        elif isinstance(dist, list):
            assert len(dist) == 2, 'Can only plot up to 2 parameters'
            for col in dist:
                assert col in self.header
            if label is None:
                x_label = dist[0]
                y_label = dist[1]
            else:
                x_label = label[0]
                y_label = label[1]                
            fig, ax = plotting.hist2d(
                self.objs[dist[0]],
                self.objs[dist[1]],
                title = "{} vs {} distribution: {}".format(dist[0], dist[1], self.name),
                show = False,
                xlabel = x_label,
                ylabel = y_label,
                logx = logx,
                logy = logy,
                log_freq = log_freq,
            )
        else:
            raise Exception('Cannot perform plot of "{}".'.format(dist))

        return fig, ax


#python 2.7 compliance
Population.next = Population.__next__

