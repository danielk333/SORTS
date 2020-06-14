#!/usr/bin/env python

'''
'''
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--orekit_data", action="store", default=None, help="Path to the orekit data archive"
    )


@pytest.fixture(scope="class")
def orekit_data(request):
    request.cls.orekit_data = request.config.getoption("--orekit_data")