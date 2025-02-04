import setuptools
from distutils.core import Extension
import pathlib
import codecs

HERE = pathlib.Path(__file__).resolve().parents[0]


def get_version(path):
    with codecs.open(path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


clibsorts = Extension(
    name="sorts.clibsorts",
    sources=[
        "src/clibsorts/radar_controller.c",
        "src/clibsorts/static_priority_scheduler.c",
        "src/clibsorts/plotting_controls.c",
        "src/clibsorts/signals.c",
        "src/clibsorts/measurements.c",
        "src/clibsorts/radar.c",
    ],
    include_dirs=[
        "src/clibsorts/",
    ],
)

setuptools.setup(
    version=get_version(HERE / "src" / "sorts" / "version.py"),
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    ext_modules=[
        clibsorts,
    ],
    package_data={
        "sorts": [
            "data/*",
        ],
    },
)
