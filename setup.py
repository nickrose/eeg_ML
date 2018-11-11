# -*- coding: utf-8 -*-
"""Setup package."""
# See https://packaging.python.org/distributing/
# and https://github.com/pypa/sampleproject/blob/master/setup.py
__version__, __author__, __email__ = ('0.1', 'Nick Roseveare',
    'nicholasroseveare@gmail.com')
try:
    from setuptools import setup, find_packages
    pkgs = find_packages()
except ImportError:
    from distutils.core import setup
    pkgs = ['eeg_project']

# with open('README.md') as f:
#         readme = f.read()

setup(name='eeg_project',
      version=__version__,
      description=('Processing, plotting, and identification of EEG signals'),
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
      )
