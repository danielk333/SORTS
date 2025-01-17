#!/usr/bin/env python

"""package data

"""
import importlib

DATA = {}

# To be compatible with 3.7-8
# as resources.files was introduced in 3.9
if hasattr(importlib.resources, "files"):
    _data_folder = importlib.resources.files("pyant.beams.data")
    for file in _data_folder.iterdir():
        if not file.is_file():
            continue
        if file.name.endswith(".py"):
            continue

        DATA[file.name] = file

else:
    _data_folder = importlib.resources.contents("pyant.beams.data")
    for fname in _data_folder:
        with importlib.resources.path("pyant.beams.data", fname) as file:
            if not file.is_file():
                continue
            if file.name.endswith(".py"):
                continue

            DATA[file.name] = pathlib.Path(str(file))
