#!/usr/bin/env python

'''CCSDS TDM/OEM file input outputs

https://public.ccsds.org/Pubs/503x0b1c1.pdf
https://public.ccsds.org/Pubs/505x0b1.pdf

'''
import pkg_resources
import pathlib
import time
import datetime
import os

from lxml import etree
import xmlschema

import numpy as np
import scipy.constants as consts
from astropy.time import TimeISO, Time


from .. import dates
from ..version import __version__



class epochType(TimeISO):
    '''Convert astropy time to "yyyy-dddThh:mm:ss.d->dZ" CCSDS 2.0 epochType standard.'''
    name = 'CCSDS_epoch'  # Unique format name
    subfmts = (
        (
            'date_hms',
            '%Y-%jT%H:%M:%S',
            '{year:d}-{yday:03d}T{hour:02d}:{min:02d}:{sec:02d}',
        ),
    )


def write_txt_tdm():
    raise NotImplementedError()
def read_txt_tdm():
    raise NotImplementedError()
def read_xml_tdm():
    raise NotImplementedError()
def write_txt_oem():
    raise NotImplementedError()
def read_txt_oem():
    raise NotImplementedError()
def write_xml_oem():
    raise NotImplementedError()
def read_xml_oem():
    raise NotImplementedError()



# needs only to be unique - does not matter what it points to
# blue book examples for both tdm versions 1 and version 2 have this,
# (even though it points to what appears to be a version 1 specific resource)
# either way, that uri does not resolve

SCHEMA_URI = 'http://sanaregistry.org/r/ndmxml/ndmxml-1.0-master.xsd'


_TDM_SCHEMA = None

def get_tdm_schema():
    global _TDM_SCHEMA
    if _TDM_SCHEMA is None:
        data_path = pkg_resources.resource_filename('sorts', 'data')
        xsd_path = os.path.join(data_path, 'ndmxml-2.0.0-master-2.0.xsd')
        _TDM_SCHEMA = xmlschema.XMLSchema(xsd_path)
    return _TDM_SCHEMA


TDM_OBSERVATION_METADATA_FIELDS = [
    "COMMENT",
    "TRACK_ID",
    "TRACK_ID",
    "DATA_TYPES",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    'PARTICIPANT_1',
    'PARTICIPANT_2',
    'PARTICIPANT_3',
    'PARTICIPANT_4',
    'PARTICIPANT_5',
    "MODE",
    "PATH",
    "EMPHEMERIS_NAME_1",
    "EMPHEMERIS_NAME_2",
    "EMPHEMERIS_NAME_3",
    "EMPHEMERIS_NAME_4",
    "EMPHEMERIS_NAME_5",
    "TRANSMIT_BAND",
    "RECEIVE_BAND",
    "TURNAROUND_NUMENATOR",
    "TURNAROUND_DENUMENATOR",
    "TIMETAG_REF",
    "INTEGRATION_INTERVAL",
    "INTEGRATION_REF",
    "FREQ_OFFSET",
    "RANGE_MODE",
    "RANGE_MODULUS",
    "RANGE_UNITS",
    "ANGLE_TYPE",
    "REFERENCE_FRAME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
    "DOPPLER_COUNT_BIAS",
    "DOPPLER_COUNT_SCALE",
    "DOPPLER_COUNT_ROLLOVER",
    "TRANSMIT_DELAY_1",
    "TRANSMIT_DELAY_2",
    "TRANSMIT_DELAY_3",
    "TRANSMIT_DELAY_4",
    "TRANSMIT_DELAY_5",
    "RECEIVE_DELAY_1",
    "RECEIVE_DELAY_2",
    "RECEIVE_DELAY_3",
    "RECEIVE_DELAY_4",
    "RECEIVE_DELAY_5",
    "DATA_QUALITY",
    "CORRECTION_ANGLE_1",
    "CORRECTION_ANGLE_2",
    "CORRECTION_DOPPLER",
    "CORRECTION_MAG",
    "CORRECTION_RANGE",
    "CORRECTION_RCS",
    "CORRECTION_RECEIVE",
    "CORRECTION_TRANSMIT",
    "CORRECTION_ABERRATION_YEARLY",
    "CORRECTION_ABERRATION_DIURNAL",
    "CORRECTIONS_APPLIED"
]


def write_xml_tdm(data, meta, file=None):

    # originator
    originator = meta.get("ORIGINATOR", f'SORTS {__version__}')

    # creation data
    creation_date = meta.get("CREATION_DATE", Time.now()).CCSDS_epoch

    # tdm dictionary
    d = {
        '@id': 'CCSDS_TDM_VERS',
        '@version': '2.0',
        '@xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        '@xsi:noNamespaceSchemaLocation': SCHEMA_URI,
        'header': {
            'CREATION_DATE': f'{creation_date}', 
            'ORIGINATOR': f'{originator}'
        },
        'body': {
            'segment': []
        }
    }

    # add segments
    d['body']['segment']

    # ALTERNATIVES
    # 
    # 1) - make one segment per observation or
    # 2) - one segment with multiple observations
    #
    # CHOOSE 2)
    segment = {
        "metadata": {},
        "data": {}
    }
    d['body']['segment'].append(segment)    

    ###########################################################################
    # SEGMENT METADATA
    ###########################################################################

    _meta = segment["metadata"]

    # mandatory
    meta_defaults = {
        "TIME_SYSTEM": "UTC",
        "PARTICIPANT_1": "missing"
    }
    
    # add values from given meta - in correct order
    for field in TDM_OBSERVATION_METADATA_FIELDS:
        if field in meta:
            _meta[field] = meta[field]
        else:
            if field in meta_defaults:
                _meta[field] = meta_defaults[field]

    ###########################################################################
    # SEGMENT DATA
    ###########################################################################

    _data = segment["data"]
    _data["COMMENT"] = 'DATA COMMENT'
    _data["observation"] = []

    for entry in data:
        _data["observation"].append({
            "EPOCH": Time(entry["EPOCH"], scale="tai", format="datetime64").CCSDS_epoch,
            "RANGE": entry['RANGE'].item()
        })

    # serialize to xml
    schema = get_tdm_schema()
    xml_etree = schema.encode(d, path='/tdm')
    
    if file is not None:
        file.write(xmlschema.etree_tostring(xml_etree, encoding='unicode', method='xml'))

    return xml_etree







# def write_oem(t, state, meta, fname=None):
#     '''Uses a series of astropy times and state vectors to create a CCSDS OEM file (plain-text of xml) or return a string.

#     :param astropy.Time t: Vector of unix-times
#     :param numpy.ndarray state: 6-D states given in SI units in the ITRF2000 frame. Rows correspond to different states and columns to dimensions.
#     :param dict meta: Dict containing all the standard CCSDS 2.0 meta data.
#     :param bool xml: If `False`, use the plain-text format
#     :param str fname: Output file-path for OEM.
#     '''
#     if xml:
#         raise NotImplementedError('')
    
#     fo.write("CCSDS_OEM_VERS = 2.0\n")
#     fo.write("CREATION_DATE = %s\n"%(dates.unix_to_datestr(t.min()))) # 1996-11-04T17:22:31
#     fo.write("ORIGINATOR = SORTS\n")
#     fo.write("META_START\n")
#     fo.write("OBJECT_NAME = %s\n"%(oid))
#     fo.write("OBJECT_ID = %d\n"%(oid))
#     fo.write("CENTER_NAME = EARTH\n")
#     fo.write("REF_FRAME = ITRF2000\n")
#     fo.write("TIME_SYSTEM = UTC\n")
#     fo.write("START_TIME = %s\n"%(dates.unix_to_datestrf(t.min())))
#     fo.write("USEABLE_START_TIME = %s\n"%(dates.unix_to_datestrf(t.min())))
#     fo.write("USEABLE_STOP_TIME = %s\n"%(dates.unix_to_datestrf(t.max())))
#     fo.write("STOP_TIME = %s\n"%(dates.unix_to_datestrf(t.max())))
#     fo.write("META_STOP\n")
#     fo.write("COMMENT This file was produced by SORTS.\n")
#     fo.write("COMMENT.\n")
#     for ti in range(len(t)):
#         fo.write("%s %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f\n"%(dates.unix_to_datestrf(t[ti]),state[ti,0],state[ti,1],state[ti,2],state[ti,3],state[ti,4],state[ti,5]))
    
#     if fname is not None:

#         fo=open(fname,"w")
#         fo.close()


# def read_oem(fname, xml=True):
#     '''Todo: docstring
#     '''
#     if xml:
#         parser = etree.XMLParser(schema = _get_oem_schema())
#         tree = etree.parse(open(fname, 'r'), parser)
#         return tree

#     else:
#         meta = {'COMMENT': ''}

#         _dtype = [
#             ('date', 'datetime64[us]'),
#             ('x', 'float64'),
#             ('y', 'float64'),
#             ('z', 'float64'),
#             ('vx', 'float64'),
#             ('vy', 'float64'),
#             ('vz', 'float64'),
#         ]

#         raw_data = []
#         DATA_ON = False
#         META_ON = False
#         with open(fname, 'r') as f:
#             for line in f:

#                 if META_ON:
#                     if line.strip() == 'META_STOP':
#                         META_ON = False
#                         DATA_ON = True
#                         continue

#                     _tmp = [x.strip() for x in line.split('=')]
#                     meta[_tmp[0]] = _tmp[1]
#                 elif DATA_ON:
#                     if line[:7] == 'COMMENT':
#                         meta['COMMENT'] += line[7:]
#                     else:
#                         raw_data.append(line.split(' '))
#                 else:
#                     if line.strip() == 'META_START':
#                         META_ON = True
#                         continue
#                     _tmp = [x.strip() for x in line.split('=')]
#                     meta[_tmp[0]] = _tmp[1]


#         data_len = len(raw_data)

#         data = np.empty((data_len, ), dtype=_dtype)

#         for ind, row in enumerate(raw_data):
#             rown = 0
#             for col, dtype in _dtype:
#                 data[ind][col] = row[rown]
#                 rown += 1

#         return data, meta



# def write_tdm(
#         datas,
#         meta,
#         fname=None,
#     ):
#     '''
#     # TODO: Document function.
#     # TODO: Update function to be more general
#     # TODO: add arbitrary fields and meta data
#     '''
#     segments = xml_tdm_segmet(datas)
#     root = xml_tdm_tree(meta, segments)

#     fo=open(fname,"w")
#     fo.write("CCSDS_TDM_VERS = 1.0\n")
#     fo.write("   COMMENT MASTER ID %s\n"%(oid))
#     fo.write("   COMMENT TX_ECEF (%1.12f,%1.12f,%1.12f)\n"%(tx_ecef[0],tx_ecef[1],tx_ecef[2]))
#     fo.write("   COMMENT RX_ECEF (%1.12f,%1.12f,%1.12f)\n"%(rx_ecef[0],rx_ecef[1],rx_ecef[2]))
#     fo.write("   COMMENT This is a simulated %s of MASTER ID %s with EISCAT 3D\n"%(tdm_type,oid))
#     fo.write("   COMMENT 233 MHz, time of flight, with ionospheric corrections\n")
#     fo.write("   COMMENT EISCAT 3D coordinates: \n")
#     fo.write("   COMMENT Author(s): SORTS.\n")
#     fo.write("   CREATION_DATE       = %s\n"%(dates.unix_to_datestr(time.time())))
#     fo.write("META_START\n")
#     fo.write("   TIME_SYSTEM         = UTC\n")
#     fo.write("   START_TIME          = %s\n"%(dates.unix_to_datestrf(t_pulse.min())))
#     fo.write("   STOP_TIME           = %s\n"%(dates.unix_to_datestrf(t_pulse.max())))
#     fo.write("   PARTICIPANT_1       = %s\n"%(tx_name))
#     fo.write("   PARTICIPANT_2       = %s\n"%(oid))
#     fo.write("   PARTICIPANT_3       = %s\n"%(rx_name))
#     fo.write("   MODE                = SEQUENTIAL\n")
#     fo.write("   PATH                = 1,2,3\n")
#     fo.write("   TRANSMIT_BAND       = %1.5f\n"%(freq))
#     fo.write("   RECEIVE_BAND        = %1.5f\n"%(freq))
#     fo.write("   TIMETAG_REF         = TRANSMIT\n")
#     fo.write("   INTEGRATION_REF     = START\n")
#     fo.write("   RANGE_MODE          = CONSTANT\n")
#     fo.write("   RANGE_MODULUS       = %1.2f\n"%(128.0*20e-3))
#     fo.write("   RANGE_UNITS         = KM\n")
#     fo.write("   DATA_QUALITY        = VALIDATED\n")
#     fo.write("   CORRECTION_RANGE    = 0.0\n")
#     fo.write("   NOISE_DATA          = ON\n")
#     fo.write("   CORRECTIONS_APPLIED = NO\n")
#     fo.write("META_STOP\n")
#     fo.write("DATA_START\n")
#     for ri in range(len(t_pulse)):
#         fo.write("   RANGE                   = %s %1.12f %1.12f\n"%(dates.unix_to_datestrf(t_pulse[ri]),m_range[ri], m_range_std[ri]))
#         fo.write("   DOPPLER_INSTANTANEOUS   = %s %1.12f %1.12f\n"%(dates.unix_to_datestrf(t_pulse[ri]),m_range_rate[ri], m_range_rate_std[ri]))
#     fo.write("DATA_STOP\n")
#     fo.close()


# def read_tdm(fname, xml=True):
#     '''Just get the range data # TODO: the rest

#     # TODO: Document function.
#     # TODO: Update plain text function to be more general and conform to CCSDS 2.0
#     '''

#     if xml:
#         parser = etree.XMLParser(schema = _get_tdm_schema())
#         tree = etree.parse(open(fname, 'r'), parser)
#         return tree

#     else:

#         meta = {'COMMENT': ''}

#         RANGE_UNITS = 'km'
#         with open(fname, 'r') as f:
#             DATA_ON = False
#             META_ON = False
#             data_raw = {}
#             for line in f:
#                 if line.strip() == 'DATA_STOP':
#                     break

#                 if META_ON:
#                     tmp_lin = line.split('=')
#                     if len(tmp_lin) > 1:
#                         meta[tmp_lin[0].strip()] = tmp_lin[1].strip()
#                         if tmp_lin[0].strip() == 'RANGE_UNITS':
#                             RANGE_UNITS = tmp_lin[1].strip().lower()
#                 elif DATA_ON:
#                     name, tmp_dat = line.split('=')

#                     name = name.strip().lower()
#                     tmp_dat = tmp_dat.strip().split(' ')

#                     if name in data_raw:
#                         data_raw[name].append(tmp_dat)
#                     else:
#                         data_raw[name] = [tmp_dat]
#                 else:
#                     if line.lstrip()[:7] == 'COMMENT':
#                         meta['COMMENT'] += line.lstrip()[7:]
#                     else:
#                         tmp_lin = line.split('=')
#                         if len(tmp_lin) > 1:
#                             meta[tmp_lin[0].strip()] = tmp_lin[1].strip()

#                 if line.strip() == 'META_START':
#                     META_ON = True
#                 if line.strip() == 'DATA_START':
#                     META_ON = False
#                     DATA_ON = True
#         _dtype = [
#             ('date', 'datetime64[us]'),
#         ]

#         data_len = len(data_raw[data_raw.keys()[0]])

#         for name in data_raw:
#             _dtype.append( (name, 'float64') )
#             _dtype.append( (name + '_err', 'float64') )

#         data = np.empty((data_len, ), dtype=_dtype)

#         date_set = False
#         for name, series in data_raw.items():
#             for ind, val in enumerate(series):
#                 if not date_set:
#                     data[ind]['date'] = np.datetime64(val[0],'us')

#                 data[ind][name] = np.float64(val[1])
#                 if len(val) > 2:
#                     data[ind][name + '_err'] = np.float64(val[2])
#                 else:
#                     data[ind][name + '_err'] = 0.0

#                 if name == 'range':
#                     if RANGE_UNITS == 's':
#                         data[ind][name] *= consts.c*1e-3

#             date_set = True

#         return data, meta
