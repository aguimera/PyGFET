#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:30:42 2017

@author: aguimera
"""

#  Copyright 2017 Carlos Pascual-Izarra <cpascual@users.sourceforge.net>
#
#  This file is part of PeakEvo.
#
#  PeakEvo is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PeakEvo is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.


from setuptools import setup, find_packages

_version = '0.1.0-alpha'

long_description = """
                   GUI for acquiring series of spectra and tracking a peak
                   """

install_requires = ['numpy',
                    'PyQt5',
                    'scipy',
                    'matplotlib',
                    'pickle']

console_scripts = ['PyDBView = peakevo.maingui:main',
                   ]

entry_points = {
                'console_scripts': console_scripts,
                }

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Gui',
    'Environment :: X11 Applications :: Qt',
    'Environment :: Win32 (MS Windows)',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: User Interfaces',
]

setup(name="peakevo",
      version=_version,
      description="Spectra acquisition application",
      long_description=long_description,
      author="Carlos Pascual-Izarra",
      author_email="cpascual@users.sourceforge.net",
      maintainer="Carlos Pascual-Izarra",
      maintainer_email="cpascual@users.sourceforge.net",
      url="https://github.com/cpascual/peakevo",
      download_url="https://github.com/cpascual/peakevo",
      license="GPLv3",
      packages=find_packages(),
      classifiers=classifiers,
      include_package_data=True,
      entry_points=entry_points,
      install_requires=install_requires
)


