#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:41:08 2017

@author: aguimera
"""

import matplotlib.pyplot as plt


def SaveOpenSigures(Dir, Prefix, OutFigFormats=('svg', 'png')):
    for i in plt.get_fignums():
        plt.figure(i)
        for ext in OutFigFormats:
            fileOut = Dir + '/' + Prefix + '{}.' + ext
            plt.savefig(fileOut.format(i))
