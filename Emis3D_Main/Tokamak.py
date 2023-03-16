#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu
"""

from os.path import dirname, realpath, join

from matplotlib import path

import numpy as np
import random
import math

FILE_PATH = dirname(realpath(__file__))
EMIS3D_PARENT_DIRECTORY = dirname(FILE_PATH)
EMIS3D_INPUTS_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "Emis3D_Inputs")

class Tokamak(object):
    
    def __init__(self, TokamakName = "JET", WallFile=None):
        
        # Angle conventions used in each tokamak are different from that used in Cherab.
        # Emis3D uses the Cherab angle convention. This angle is subtracted in the evaluate
        # statements in RadDist to make the angles match.
        torConventionPhis = {"JET": math.pi / 2.0,
            "SPARC": 0.0}#math.pi / 2.0}
        
        self.torConventionPhi = torConventionPhis[TokamakName]
        self.wallfile = WallFile
        self.tokamakName = TokamakName

# loads the first wall path from file
    def load_first_wall(self):
        try:
            rzarray = np.loadtxt(self.wallfile, skiprows=0)
        except:
            rzarray = np.loadtxt(self.wallfile, delimiter=',', skiprows=0)

        rzarray = np.array(rzarray)

        
        minr = min(rzarray[:,0])
        maxr = max(rzarray[:,0])
        minz = min(rzarray[:,1])
        maxz = max(rzarray[:,1])
        wallcurve = path.Path(rzarray)
        return wallcurve, minr, maxr, minz, maxz

    def volume_from_first_wall(self, Numpoints):

        def random_uniform_point(Wallcurve,\
            Minr, Maxr, Minz, Maxz):
            
            inVolume = False
            
            x = random.uniform(-Maxr, Maxr)
            y = random.uniform(-Maxr, Maxr)
            z = random.uniform(Minz, Maxz)
            r = np.sqrt((x**2)+(y**2))
            
            if (r < Minr or r > Maxr):
                pass
            elif Wallcurve.contains_points([(r, z)]):
                inVolume = True
                    
            return inVolume

        numpoints = 0
        pointsInVolume = 0
        while numpoints < Numpoints:
            inVolume\
                = random_uniform_point(Wallcurve=self.wallcurve,\
                    Minr=self.minr, Maxr=self.maxr,\
                    Minz=self.minz, Maxz=self.maxz)

            if inVolume:
                pointsInVolume = pointsInVolume + 1

            numpoints = numpoints + 1

        volumeFraction = pointsInVolume / Numpoints
        print(volumeFraction)

        rectVolume = (2.0 * self.maxr)**2 * (self.maxz - self.minz)
        volume = rectVolume * volumeFraction

        return volume
