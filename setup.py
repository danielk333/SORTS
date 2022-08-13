import setuptools
from distutils.core import Extension

clibsorts = Extension(
    name='sorts.clibsorts',
    sources=[
        'src/clibsorts/radar_controller.c',
        'src/clibsorts/static_priority_scheduler.c',
        'src/clibsorts/plotting_controls.c',
        'src/clibsorts/signals.c',
        'src/clibsorts/measurements.c',
        'src/clibsorts/radar.c',
    ],
    include_dirs=[
        'src/clibsorts/',
    ],
)

setuptools.setup(
    package_dir = {
        "": "src"
    },
    packages=setuptools.find_packages(where="src"),
    ext_modules = [
        clibsorts,
    ],
    package_data = {
        'sorts': [
            'data/*',
        ],
    },
)
