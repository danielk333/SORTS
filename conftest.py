#!/usr/bin/env python

"""
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--orekit_data", action="store", default=None, help="Path to the orekit data archive"
    )
    parser.addoption(
        "--skipslow",
        action="store_true",
        default=False,
        help="skip slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


@pytest.fixture(scope="class")
def orekit_data(request):
    request.cls.orekit_data = request.config.getoption("--orekit_data")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skipslow"):
        # --skipslow not given in cli: run all tests
        return

    skip_slow = pytest.mark.skip(reason="--skipslow option used")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
