#!/usr/bin/env python

'''CCSDS TDM/OEM file input outputs

Orbit Data Messages  
ODM     502.0-B-2   Blue Book (11/2009)  
https://public.ccsds.org/Pubs/502x0b2c1e2.pdf

Tracking Data Message
TDM     503.0-B-2   Blue Book (06/2020) 
https://public.ccsds.org/Pubs/503x0b2.pdf

NPM (Navigation Data Messages)
https://public.ccsds.org/Pubs/505x0b1.pdf

NPM Schemas covering TDM and OEM and more
https://sanaregistry.org/files/ndmxml_unqualified/*
https://sanaregistry.org/files/ndmxml_qualified/*


'''

import pkg_resources
# import time
# import datetime
import os
import xmlschema
# import numpy as np
# import scipy.constants as consts
from astropy.time import TimeISO, Time
# from .. import dates
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


# needs only to be unique - does not matter what it points to
# blue book examples for both tdm versions 1 and version 2 have this,
# (even though it points to what appears to be a version 1 specific resource)
# either way, that uri does not resolve

SCHEMA_URI = 'http://sanaregistry.org/r/ndmxml/ndmxml-1.0-master.xsd'



###############################################################################
# NDM SCHEMA
###############################################################################

_SCHEMA = None

def get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        data_path = pkg_resources.resource_filename('sorts', 'data')
        xsd_path = os.path.join(data_path, 'ndmxml-2.0.0-master-2.0.xsd')
        _SCHEMA = xmlschema.XMLSchema(xsd_path)
    return _SCHEMA


###############################################################################
# OEM
###############################################################################


_OEM_METADATA_FIELDS = [
    "COMMENT",
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "START_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE"
]

###############################################################################
# OEM READ
###############################################################################

def read_xml_oem(xml):
    """
    read xml OEM file resource, parse and validate according to schema,
    returns python dictionary representing document
    """
    schema = get_schema()
    d = schema.decode(xml)
    return d


###############################################################################
# OEM WRITE
###############################################################################

def write_xml_oem(data, meta, file=None):    
    """
    creaate and validate xml for OEM message
    write xml file resource
    """
    # originator
    originator = meta.get("ORIGINATOR", f'SORTS {__version__}')

    # creation data
    creation_date = meta.get("CREATION_DATE", Time.now()).CCSDS_epoch

    # tdm dictionary
    d = {
        '@id': 'CCSDS_OEM_VERS',
        '@version': '2.0',
        '@xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        '@xsi:noNamespaceSchemaLocation': SCHEMA_URI,
        'header': {
            'COMMENT': "HEADER COMMENT",
            'CREATION_DATE': f'{creation_date}', 
            'ORIGINATOR': f'{originator}'
        },
        'body': {
            'segment': []
        }
    }

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
    meta_defaults = dict(
        OBJECT_NAME = "missing",
        OBJECT_ID = "missing",
        CENTER_NAME = "missing",
        REF_FRAME = "missing",
        TIME_SYSTEM = "UTC",
        START_TIME = Time.now().CCSDS_epoch,
        STOP_TIME = Time.now().CCSDS_epoch,
    )
    
    # add values from given meta - in correct order
    for field in _OEM_METADATA_FIELDS:
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
    _data["stateVector"] = []

    mock_vectors = [
        [Time.now().CCSDS_epoch, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [Time.now().CCSDS_epoch, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
    ]

    for v in mock_vectors:
        _data["stateVector"].append(dict(
            EPOCH = v[0],
            X = v[1],
            Y = v[2],
            Z = v[3],
            X_DOT = v[4],
            Y_DOT = v[5],
            Z_DOT = v[6],
            X_DDOT = v[7],
            Y_DDOT = v[8],
            Z_DDOT = v[9]
        ))


    mock_matrices = [
        [1.1]*21,
        [2.2]*21
    ]

    _data["covarianceMatrix"] = []
    for m in mock_matrices:
        _data["covarianceMatrix"].append(dict(
            COMMENT = "COMMENT",
            EPOCH = Time.now().CCSDS_epoch,
            COV_REF_FRAME= "missing",
            CX_X = m[0],
            CY_X = m[1],
            CY_Y = m[2],
            CZ_X = m[3],
            CZ_Y = m[4],
            CZ_Z = m[5],
            CX_DOT_X = m[6],
            CX_DOT_Y = m[7],
            CX_DOT_Z = m[8],
            CX_DOT_X_DOT = m[9],
            CY_DOT_X = m[10],
            CY_DOT_Y = m[11],
            CY_DOT_Z = m[12],
            CY_DOT_X_DOT = m[13],
            CY_DOT_Y_DOT = m[14],
            CZ_DOT_X = m[15],
            CZ_DOT_Y = m[16],
            CZ_DOT_Z = m[17],
            CZ_DOT_X_DOT = m[18],
            CZ_DOT_Y_DOT = m[19],
            CZ_DOT_Z_DOT = m[20]
        ))
    
    # serialize to xml
    schema = get_schema()
    xml_etree = schema.encode(d, path='/oem')
    
    if file is not None:
        file.write(xmlschema.etree_tostring(xml_etree, encoding='unicode', method='xml'))

    return xml_etree


###############################################################################
# TDM
###############################################################################

_TDM_METADATA_FIELDS = [
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


###############################################################################
# TDM READ
###############################################################################


def read_xml_tdm(xml):
    """
    read xml TMD file resource, parse and validate according to schema,
    returns python dictionary representing document
    """
    schema = get_schema()
    d = schema.decode(xml)
    return d


###############################################################################
# TDM WRITE
###############################################################################

def write_xml_tdm(data, meta, file=None):
    """
    creaate and validate xml for TDM message
    write xml file resource
    """
    # originator
    originator = meta.get("ORIGINATOR", f'SORTS {__version__}')

    # creation data
    creation_date = meta.get("CREATION_DATE", Time.now()).CCSDS_epoch

    # message id
    message_id = meta.get("MESSAGE_ID", 0)

    # tdm dictionary
    d = {
        '@id': 'CCSDS_TDM_VERS',
        '@version': '2.0',
        '@xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        '@xsi:noNamespaceSchemaLocation': SCHEMA_URI,
        'header': {
            'CREATION_DATE': f'{creation_date}', 
            'ORIGINATOR': f'{originator}',
            'MESSAGE_ID': f'{message_id}'
        },
        'body': {
            'segment': []
        }
    }

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
    for field in _TDM_METADATA_FIELDS:
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
    schema = get_schema()
    xml_etree = schema.encode(d, path='/tdm')
    
    if file is not None:
        file.write(xmlschema.etree_tostring(xml_etree, encoding='unicode', method='xml'))

    return xml_etree

