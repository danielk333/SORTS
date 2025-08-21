#!/usr/bin/env python

'''
Writing CCSDS files
================================

'''
import io

import numpy as np
from astropy.time import Time, TimeDelta

import sorts
radar = sorts.radars.eiscat3d

np.random.seed(0)

rows = 10

meta = dict(
    COMMENT = 'This is a test',
)
data_tdm = np.empty(
    (rows,), 
    dtype=[
        ('EPOCH', 'datetime64[ns]'),
        ('RANGE', 'f8'),
        ('DOPPLER_INSTANTANEOUS', 'f8'),
    ],
)
data_tdm['RANGE'] = np.random.randn(rows)
data_tdm['DOPPLER_INSTANTANEOUS'] = np.random.randn(rows)
data_tdm['EPOCH'] = (Time('J2000') + TimeDelta(np.linspace(0,40,num=rows), format='sec')).datetime64

stream = io.StringIO()

sorts.io.ccsds.write_xml_tdm(data_tdm, meta, file=stream)

print('XML output content:')
print(stream.getvalue())

stream.close()


data_oem = np.empty(
    (rows,), 
    dtype=[
        ('EPOCH', 'datetime64[ns]'),
        ('X', 'f8'),
        ('Y', 'f8'),
        ('Z', 'f8'),
        ('X_DOT', 'f8'),
        ('Y_DOT', 'f8'),
        ('Z_DOT', 'f8'),
    ],
)
data_oem['EPOCH'] = (Time('J2000') + TimeDelta(np.linspace(0,40,num=rows), format='sec')).datetime64
for key in ['X','Y','Z','X_DOT','Y_DOT','Z_DOT']:
    data_oem[key] = np.random.randn(rows)



data_oem_cov = np.empty(
    (1,), 
    dtype=[
        ('EPOCH', 'datetime64[ns]'),
        ('COMMENT', 'S8'),
        ('CX_X', 'f8'),
        ('CY_X', 'f8'),
        ('CY_Y', 'f8'),
        ('CZ_X', 'f8'),
        ('CZ_Y', 'f8'),
        ('CZ_Z', 'f8'),
        ('CX_DOT_X', 'f8'),
        ('CX_DOT_Y', 'f8'),
        ('CX_DOT_Z', 'f8'),
        ('CX_DOT_X_DOT', 'f8'),
        ('CY_DOT_X', 'f8'),
        ('CY_DOT_Y', 'f8'),
        ('CY_DOT_Z', 'f8'),
        ('CY_DOT_X_DOT', 'f8'),
        ('CY_DOT_Y_DOT', 'f8'),
        ('CZ_DOT_X', 'f8'),
        ('CZ_DOT_Y', 'f8'),
        ('CZ_DOT_Z', 'f8'),
        ('CZ_DOT_X_DOT', 'f8'),
        ('CZ_DOT_Y_DOT', 'f8'),
        ('CZ_DOT_Z_DOT', 'f8'),
    ],
)

data_oem_cov[0]['COMMENT'] = 'wow'
for key in data_oem_cov.dtype.names:
    if key not in ['COMMENT', 'EPOCH']:
        data_oem_cov[key] = np.random.randn(1)
data_oem_cov['EPOCH'] = Time('J2000').datetime64

stream = io.StringIO()

sorts.io.ccsds.write_xml_oem(data_oem, data_oem_cov, meta, file=stream)

print('OEM output content:')
print(stream.getvalue())

stream.close()