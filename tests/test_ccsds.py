import unittest
import sorts
import io
import numpy as np
from astropy.time import Time, TimeDelta


class CCSDSTest(unittest.TestCase):


    def test_tdm(self):

        radar = sorts.radars.eiscat3d
        np.random.seed(0)
        rows = 10

        meta = dict(
            COMMENT = 'This is a test',
        )
        data = np.empty(
            (rows,), 
            dtype=[
                ('EPOCH', 'datetime64[ns]'),
                ('RANGE', 'f8'),
                ('DOPPLER_INSTANTANEOUS', 'f8'),
            ],
        )
        data['RANGE'] = np.random.randn(rows)
        data['DOPPLER_INSTANTANEOUS'] = np.random.randn(rows)
        data['EPOCH'] = (Time('J2000') + TimeDelta(np.linspace(0,40,num=rows), format='sec')).datetime64

        stream = io.StringIO()

        sorts.io.ccsds.write_xml_tdm(data, meta, file=stream)

        print('XML output content:')
        print(stream.getvalue())

        stream.close()
