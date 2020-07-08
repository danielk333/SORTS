import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

import sorts.version

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    

    def run(self):

        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    

    def run(self):

        install.run(self)


with open('README.rst', 'r') as fh:
    long_description = fh.read()


with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]


setuptools.setup(
    name='sorts',
    version=sorts.version.__version__,
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
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
