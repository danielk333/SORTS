import unittest
import sorts
import io
import numpy as np

from astropy.time import Time, TimeDelta
import pyant

class CCSDSTest(unittest.TestCase):
    def test_write_oem(self):

        stream = io.StringIO()
        meta = dict(
            COMMENT = 'This is a test',
        )

        # test data
        data = np.empty((1, ), dtype=_dtype)
        _dtype = [
            ('date', 'datetime64[us]'),
            ('x', 'float64'),
            ('y', 'float64'),
            ('z', 'float64'),
            ('vx', 'float64'),
            ('vy', 'float64'),
            ('vz', 'float64'),
        ]

        raw_data = np.linspace(1, 10, 10)
        raw_data[0] = Time("J2000").datetime64()
        ind = 0
        for name_i, dtype_i in _dtype:
            data[name_i][0] = raw_data[ind]
            ind += 1

        sorts.io.ccsds.write_xml_oem(data, meta=meta, file=stream)
        xml = stream.getvalue()

        self.assertTrue(isinstance(xml, str))
        self.assertTrue(xml)

        stream.close()

    def test_read_oem(self):
        OEM_XML =   """
                    <oem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" id="CCSDS_OEM_VERS" version="2.0" xsi:noNamespaceSchemaLocation="http://sanaregistry.org/r/ndmxml/ndmxml-1.0-master.xsd">
                        <header>
                            <COMMENT>HEADER COMMENT</COMMENT>
                            <CREATION_DATE>2021-049T17:26:29.560</CREATION_DATE>
                            <ORIGINATOR>SORTS 4.0.0-rc.1</ORIGINATOR>
                        </header>
                        <body>
                            <segment>
                                <metadata>
                                    <COMMENT>This is a test</COMMENT>
                                    <OBJECT_NAME>missing</OBJECT_NAME>
                                    <OBJECT_ID>missing</OBJECT_ID>
                                    <CENTER_NAME>missing</CENTER_NAME>
                                    <REF_FRAME>missing</REF_FRAME>
                                    <TIME_SYSTEM>UTC</TIME_SYSTEM>
                                    <START_TIME>2021-049T17:26:29.560</START_TIME>
                                    <STOP_TIME>2021-049T17:26:29.560</STOP_TIME>
                                </metadata>
                                <data>
                                    <COMMENT>DATA COMMENT</COMMENT>
                                    <stateVector>
                                        <EPOCH>2021-049T17:26:29.561</EPOCH>
                                        <X>1.1</X>
                                        <Y>2.2</Y>
                                        <Z>3.3</Z>
                                        <X_DOT>4.4</X_DOT>
                                        <Y_DOT>5.5</Y_DOT>
                                        <Z_DOT>6.6</Z_DOT>
                                        <X_DDOT>7.7</X_DDOT>
                                        <Y_DDOT>8.8</Y_DDOT>
                                        <Z_DDOT>9.9</Z_DDOT>
                                    </stateVector>
                                    <stateVector>
                                        <EPOCH>2021-049T17:26:29.561</EPOCH>
                                        <X>1.1</X>
                                        <Y>2.2</Y>
                                        <Z>3.3</Z>
                                        <X_DOT>4.4</X_DOT>
                                        <Y_DOT>5.5</Y_DOT>
                                        <Z_DOT>6.6</Z_DOT>
                                        <X_DDOT>7.7</X_DDOT>
                                        <Y_DDOT>8.8</Y_DDOT>
                                        <Z_DDOT>9.9</Z_DDOT>
                                    </stateVector>
                                    <covarianceMatrix>
                                        <COMMENT>COMMENT</COMMENT>
                                        <EPOCH>2021-049T17:26:29.561</EPOCH>
                                        <COV_REF_FRAME>missing</COV_REF_FRAME>
                                        <CX_X>1.1</CX_X>
                                        <CY_X>1.1</CY_X>
                                        <CY_Y>1.1</CY_Y>
                                        <CZ_X>1.1</CZ_X>
                                        <CZ_Y>1.1</CZ_Y>
                                        <CZ_Z>1.1</CZ_Z>
                                        <CX_DOT_X>1.1</CX_DOT_X>
                                        <CX_DOT_Y>1.1</CX_DOT_Y>
                                        <CX_DOT_Z>1.1</CX_DOT_Z>
                                        <CX_DOT_X_DOT>1.1</CX_DOT_X_DOT>
                                        <CY_DOT_X>1.1</CY_DOT_X>
                                        <CY_DOT_Y>1.1</CY_DOT_Y>
                                        <CY_DOT_Z>1.1</CY_DOT_Z>
                                        <CY_DOT_X_DOT>1.1</CY_DOT_X_DOT>
                                        <CY_DOT_Y_DOT>1.1</CY_DOT_Y_DOT>
                                        <CZ_DOT_X>1.1</CZ_DOT_X>
                                        <CZ_DOT_Y>1.1</CZ_DOT_Y>
                                        <CZ_DOT_Z>1.1</CZ_DOT_Z>
                                        <CZ_DOT_X_DOT>1.1</CZ_DOT_X_DOT>
                                        <CZ_DOT_Y_DOT>1.1</CZ_DOT_Y_DOT>
                                        <CZ_DOT_Z_DOT>1.1</CZ_DOT_Z_DOT>
                                    </covarianceMatrix>
                                    <covarianceMatrix>
                                        <COMMENT>COMMENT</COMMENT>
                                        <EPOCH>2021-049T17:26:29.561</EPOCH>
                                        <COV_REF_FRAME>missing</COV_REF_FRAME>
                                        <CX_X>2.2</CX_X>
                                        <CY_X>2.2</CY_X>
                                        <CY_Y>2.2</CY_Y>
                                        <CZ_X>2.2</CZ_X>
                                        <CZ_Y>2.2</CZ_Y>
                                        <CZ_Z>2.2</CZ_Z>
                                        <CX_DOT_X>2.2</CX_DOT_X>
                                        <CX_DOT_Y>2.2</CX_DOT_Y>
                                        <CX_DOT_Z>2.2</CX_DOT_Z>
                                        <CX_DOT_X_DOT>2.2</CX_DOT_X_DOT>
                                        <CY_DOT_X>2.2</CY_DOT_X>
                                        <CY_DOT_Y>2.2</CY_DOT_Y>
                                        <CY_DOT_Z>2.2</CY_DOT_Z>
                                        <CY_DOT_X_DOT>2.2</CY_DOT_X_DOT>
                                        <CY_DOT_Y_DOT>2.2</CY_DOT_Y_DOT>
                                        <CZ_DOT_X>2.2</CZ_DOT_X>
                                        <CZ_DOT_Y>2.2</CZ_DOT_Y>
                                        <CZ_DOT_Z>2.2</CZ_DOT_Z>
                                        <CZ_DOT_X_DOT>2.2</CZ_DOT_X_DOT>
                                        <CZ_DOT_Y_DOT>2.2</CZ_DOT_Y_DOT>
                                        <CZ_DOT_Z_DOT>2.2</CZ_DOT_Z_DOT>
                                    </covarianceMatrix>
                                </data>
                            </segment>
                        </body>
                    </oem>
                    """
                    
        d = sorts.io.ccsds.read_xml_tdm(OEM_XML)
        
        self.assertTrue(isinstance(d, dict))
        self.assertTrue(d)
        
        from pprint import pprint
        # pprint(d)


    def test_write_tdm(self):
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
        sorts.io.ccsds.write_xml_tdm(data, meta=meta, file=stream)
        xml = stream.getvalue()

        self.assertTrue(isinstance(xml, str))
        self.assertTrue(xml)

        # print(xml)

        stream.close()


    def test_read_tdm(self):

        TDM_XML =   """
                    <tdm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" id="CCSDS_TDM_VERS" version="2.0" xsi:noNamespaceSchemaLocation="http://sanaregistry.org/r/ndmxml/ndmxml-1.0-master.xsd">
                        <header>
                            <CREATION_DATE>2021-049T14:14:53.488</CREATION_DATE>
                            <ORIGINATOR>SORTS 4.0.0-rc.1</ORIGINATOR>
                        </header>
                        <body>
                            <segment>
                                <metadata>
                                    <COMMENT>This is a test</COMMENT>
                                    <TIME_SYSTEM>UTC</TIME_SYSTEM>
                                    <PARTICIPANT_1>missing</PARTICIPANT_1>
                                </metadata>
                                <data>
                                    <COMMENT>DATA COMMENT</COMMENT>
                                    <observation>
                                        <EPOCH>2000-001T12:00:00.000</EPOCH>
                                        <RANGE>1.764052345967664</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:04.444</EPOCH>
                                        <RANGE>0.4001572083672233</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:08.889</EPOCH>
                                        <RANGE>0.9787379841057392</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:13.333</EPOCH>
                                        <RANGE>2.240893199201458</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:17.778</EPOCH>
                                        <RANGE>1.8675579901499675</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:22.222</EPOCH>
                                        <RANGE>-0.977277879876411</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:26.667</EPOCH>
                                        <RANGE>0.9500884175255894</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:31.111</EPOCH>
                                        <RANGE>-0.1513572082976979</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:35.556</EPOCH>
                                        <RANGE>-0.10321885179355784</RANGE>
                                    </observation>
                                    <observation>
                                        <EPOCH>2000-001T12:00:40.000</EPOCH>
                                        <RANGE>0.41059850193837233</RANGE>
                                    </observation>
                                </data>
                            </segment>
                        </body>
                    </tdm>
                    """
                    
        d = sorts.io.ccsds.read_xml_tdm(TDM_XML)
        
        self.assertTrue(isinstance(d, dict))
        self.assertTrue(d)

        # pprint(d)