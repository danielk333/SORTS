import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
import pip
import pathlib
import codecs

HERE = pathlib.Path(__file__).resolve().parents[0]


def get_version(path):
    with codecs.open(path, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


with open('dependency-links', 'r') as fh:
    dep_links = fh.read().split('\n')
    dep_links = [x.strip() for x in dep_links if len(x.strip()) > 0]

with open('README.rst', 'r') as fh:
    long_description = fh.read()

with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]

setuptools.setup(
    name='sorts',
    version=get_version(HERE / 'sorts' / 'version.py'),
    long_description=long_description,
    url='https://gitlab.irf.se/danielk/SORTS',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT',
        'Operating System :: Unix',
        'Intended Audience :: Science/Research',
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    package_data={
        'sorts': ['data/*'],
    },
    # metadata to display on PyPI
    author='Daniel Kastinen, Juha Vierinen, Tom Grydeland',
    author_email='daniel.kastinen@irf.se',
    description='SORTS',
    license='MIT',
    cmdclass={},
)
