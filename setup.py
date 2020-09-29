import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
import subprocess
import pip

__version__ = '4.0.0-beta'


with open('dependency-links', 'r') as fh:
    dep_links = fh.read().split('\n')
    dep_links = [x.strip() for x in dep_links if len(x.strip()) > 0]

with open('README.rst', 'r') as fh:
    long_description = fh.read()

with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]


class CustomDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        for link in dep_links:
            pip.main(['install', link])
        develop.run(self)

class CustomInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        for link in dep_links:
            pip.main(['install', link])
        install.run(self)

class CustomEggCommand(egg_info):
    """Post-installation for installation mode."""
    def run(self):
        for link in dep_links:
            pip.main(['install', link])
        egg_info.run(self)


setuptools.setup(
    name='sorts',
    version=__version__,
    long_description=long_description,
    url='https://gitlab.irf.se/danielk/SORTSpp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT',
        'Operating System :: OS Independent',
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    package_data={
        'sorts': ['data/*'],
    },
    # metadata to display on PyPI
    author='Daniel Kastinen, Juha Vierinen',
    author_email='daniel.kastinen@irf.se',
    description='SORTS',
    license='MIT',
    cmdclass={
        'develop': CustomDevelopCommand,
        'install': CustomInstallCommand,
        'egg_info': CustomEggCommand,
    },
)
