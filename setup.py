# -*- coding: utf-8 -*-
"""Setup package."""
# See https://packaging.python.org/distributing/
# and https://github.com/pypa/sampleproject/blob/master/setup.py
__version__, __author__, __email__ = ('0.1', 'Nick Roseveare',
    'nicholasroseveare@gmail.com')
# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    from pip.req import parse_requirements
    install_reqs = parse_requirements('./requirements.txt')
    req_kwargs = dict(install_requires=install_reqs)
except ImportError:
    import warnings
    warnings.warn('could not import tools for parsing and installing required packages, '
        'it may be that the setup.py will complete sucessfully, but that you do (or more'
        ' likely, do not have the required packages installed)')
    req_kwargs = {}
try:
    from setuptools import setup, find_packages
    pkgs = find_packages()
except ImportError:
    from distutils.core import setup
    pkgs = ['eeg_project']

with open('README.md') as f:
        readme = f.read()

setup(name='eeg_project',
      version=__version__,
      description=('Processing, plotting, and identification of EEG signals'),
      long_description=readme,
      long_description_content_type='text/markdown',
      author=__author__,
      author_email=__email__,
      url='https://github.com/nickrose',
      license='MIT',
      packages=pkgs,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      **req_kwargs
      )
