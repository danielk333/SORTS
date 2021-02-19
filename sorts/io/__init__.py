#!/usr/bin/env python

'''Provides functions for read and writing data in standardized formats such as the CCSDS standards.

'''

from .ccsds import write_xml_tdm, read_xml_tdm
from .ccsds import write_xml_oem, read_xml_oem
from .terminal import flush_table, step_flush_table