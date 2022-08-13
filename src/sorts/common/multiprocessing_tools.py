import multiprocessing as mp
import numpy as np
import os

def convert_to_shared_array(array, ctype):
	'''Converts a numpy array to an array which can be shared between multiple Processes
	'''
	shared_array = mp.Array(ctype, array.size, lock=False)

	temp = np.frombuffer(shared_array, dtype=array.dtype)
	temp[:] = array.flatten(order='C')

	return shared_array


def convert_to_numpy_array(shared_array, shape):
	'''Converts an array shared berween processes to a numpy array
	'''
	array = np.ctypeslib.as_array(shared_array)

	return array.reshape(shape)

def get_process_id():
	return os.getpid()