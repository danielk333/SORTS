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
import xmlschema
from astropy.time import TimeISO, Time
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


###############################################################################
# NDM SCHEMA
###############################################################################

# needs only to be unique - does not matter what it points to
# blue book examples for both tdm versions 1 and version 2 have this,
# (even though it points to what appears to be a version 1 specific resource)
# either way, that uri does not resolve

_SCHEMA_URI = 'http://sanaregistry.org/r/ndmxml/ndmxml-1.0-master.xsd'


_SCHEMA = None

def get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        xsd_path = pkg_resources.resource_filename('sorts.data', 'ndmxml-2.0.0-master-2.0.xsd')
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

def write_xml_oem(state_data, cov_data=None, meta=None, file=None):    
    """
    creaate and validate xml for OEM message
    write xml file resource

    #TODO: finish this docstring

    assumes UTC input times

    It is assumed that the covariance matrix reference frame is the same as the state vector reference frame, i.e. `COV_REF_FRAME = meta['REF_FRAME']`.

    :param numpy.ndarray state_data: structured numpy array where each element in the array contains the data fields with the same name as those defined by the OEM XML for states.
    :param numpy.ndarray cov_data: structured numpy array where each element in the array contains the data fields with the same name as those defined by the OEM XML for covariance matricies.
    :param dict meta: .... Only exception is `DATA_COMMENT`, which is used as input for the `COMMENT` field for the data section rather then a `DATA_COMMENT` field in the meta data section.
    """
    if meta is None:
        meta = {}
    # originator
    originator = meta.get("ORIGINATOR", f'SORTS {__version__}')

    # creation data
    creation_date = meta.get("CREATION_DATE", Time.now()).CCSDS_epoch

    # tdm dictionary
    d = {
        '@id': 'CCSDS_OEM_VERS',
        '@version': '2.0',
        '@xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        '@xsi:noNamespaceSchemaLocation': _SCHEMA_URI,
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
            #ensure correct CCSDS formatting of input astropy time object
            if field.endswith('_TIME'):
                _meta[field] = _meta[field].CCSDS_epoch
        else:
            if field in meta_defaults:
                _meta[field] = meta_defaults[field]

    ###########################################################################
    # SEGMENT DATA
    ###########################################################################

    _data = segment["data"]
    if 'DATA_COMMENT' in meta:
        _data["COMMENT"] = meta["DATA_COMMENT"]

    _data["stateVector"] = []
    _data["covarianceMatrix"] = []

    mock_vectors = [
        [Time.now().CCSDS_epoch, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [Time.now().CCSDS_epoch, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
    ]


    for v in state_data:
        _dat = dict(
            EPOCH = Time(v["EPOCH"], format="datetime64", scale="utc").CCSDS_epoch,
            X = v["X"],
            Y = v["Y"],
            Z = v["Z"],
            X_DOT = v["X_DOT"],
            Y_DOT = v["Y_DOT"],
            Z_DOT = v["Z_DOT"],
        )
        #optional
        for key in ["X_DDOT", "Y_DDOT", "Z_DDOT"]:
            if key in state_data.dtype.names:
                _dat[key] = v[key]

        _data["stateVector"].append(_dat)


    if cov_data is not None:
        for v in cov_data:
            _dat_cov = dict()
            if "COMMENT" in cov_data.dtype.names:
                try:
                    _comment = v["COMMENT"].decode()
                except AttributeError:
                    _comment = v["COMMENT"]

                _dat_cov["COMMENT"] = _comment
                #remove empty comments
                if len(_dat_cov["COMMENT"]) == 0:
                    del _dat_cov["COMMENT"]

            _dat_cov.update(dict(
                COMMENT = v["COMMENT"],
                EPOCH = Time(v["EPOCH"], format="datetime64", scale="utc").CCSDS_epoch,
                COV_REF_FRAME = _meta["REF_FRAME"],
                CX_X = v["CX_X"],
                CY_X = v["CY_X"],
                CY_Y = v["CY_Y"],
                CZ_X = v["CZ_X"],
                CZ_Y = v["CZ_Y"],
                CZ_Z = v["CZ_Z"],
                CX_DOT_X = v["CX_DOT_X"],
                CX_DOT_Y = v["CX_DOT_Y"],
                CX_DOT_Z = v["CX_DOT_Z"],
                CX_DOT_X_DOT = v["CX_DOT_X_DOT"],
                CY_DOT_X = v["CY_DOT_X"],
                CY_DOT_Y = v["CY_DOT_Y"],
                CY_DOT_Z = v["CY_DOT_Z"],
                CY_DOT_X_DOT = v["CY_DOT_X_DOT"],
                CY_DOT_Y_DOT = v["CY_DOT_Y_DOT"],
                CZ_DOT_X = v["CZ_DOT_X"],
                CZ_DOT_Y = v["CZ_DOT_Y"],
                CZ_DOT_Z = v["CZ_DOT_Z"],
                CZ_DOT_X_DOT = v["CZ_DOT_X_DOT"],
                CZ_DOT_Y_DOT = v["CZ_DOT_Y_DOT"],
                CZ_DOT_Z_DOT = v["CZ_DOT_Z_DOT"]
            ))

            _data["covarianceMatrix"].append(_dat_cov)

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

_TDM_DATA_FIELDS = [
    "ANGLE_1",
    "ANGLE_2",
    "CARRIER_POWER",
    "CLOCK_BIAS",
    "CLOCK_DRIFT",
    "DOPPLER_COUNT",
    "DOPPLER_INSTANTANEOUS",
    "DOPPLER_INTEGRATED",
    "DOR",
    "MAG",
    "PC_N0",
    "PR_N0",
    "PRESSURE",
    "RANGE",
    "RCS",
    "RECEIVE_FREQ",
    "RECEIVE_FREQ_1",
    "RECEIVE_FREQ_2",
    "RECEIVE_FREQ_3",
    "RECEIVE_FREQ_4",
    "RECEIVE_FREQ_5",
    "RECEIVE_PHASE_CT_1",
    "RECEIVE_PHASE_CT_2",
    "RECEIVE_PHASE_CT_3",
    "RECEIVE_PHASE_CT_4",
    "RECEIVE_PHASE_CT_5",
    "RHUMIDITY",
    "STEC",
    "TEMPERATURE",
    "TRANSMIT_FREQ_1",
    "TRANSMIT_FREQ_2",
    "TRANSMIT_FREQ_3",
    "TRANSMIT_FREQ_4",
    "TRANSMIT_FREQ_5",
    "TRANSMIT_FREQ_RATE_1",
    "TRANSMIT_FREQ_RATE_2",
    "TRANSMIT_FREQ_RATE_3",
    "TRANSMIT_FREQ_RATE_4",
    "TRANSMIT_FREQ_RATE_5",
    "TRANSMIT_PHASE_CT_1",
    "TRANSMIT_PHASE_CT_2",
    "TRANSMIT_PHASE_CT_3",
    "TRANSMIT_PHASE_CT_4",
    "TRANSMIT_PHASE_CT_5",
    "TROPO_DRY",
    "TROPO_WET",
    "VLBI_DELAY",
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

    assumes tai input time.
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
        '@xsi:noNamespaceSchemaLocation': _SCHEMA_URI,
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
    if 'DATA_COMMENT' in meta:
        _data["COMMENT"] = meta['DATA_COMMENT']

    _data["observation"] = []

    for entry in data:
        for key in _TDM_DATA_FIELDS:
            if key in data.dtype.names:
                _dat = {
                    "EPOCH": Time(entry["EPOCH"], scale="tai", format="datetime64").CCSDS_epoch,
                }
                _dat[key] = entry[key]
                _data["observation"].append(_dat)

    # serialize to xml
    schema = get_schema()
    xml_etree = schema.encode(d, path='/tdm')
    
    if file is not None:
        file.write(xmlschema.etree_tostring(xml_etree, encoding='unicode', method='xml'))

    return xml_etree

