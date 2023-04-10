#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:55:26 2023

@author: br0148
"""

from os.path import dirname, realpath, join
        
FILE_PATH = dirname(realpath(__file__))
EMIS3D_PARENT_DIRECTORY = dirname(dirname(FILE_PATH))
EMIS3D_UNIVERSAL_MAIN_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY,\
    "Emis3D_Universal", "Emis3D_Main")
EMIS3D_INPUTS_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "Emis3D_JET", "Emis3D_Inputs")

import numpy as np

class Emis3D(object):
    
    def __init__(self):
        
        self.allRadDistsVec = []
        
        self.tokamakAMode = None # Always loaded
        self.tokamakBMode = None # Not loaded unless necessary; time consuming
        self.load_tokamak(Mode="Analysis")
        
        self.numTorLocs = self.tokamakAMode.numTorLocs
        
        # In calc_fits
        self.chisqVec = None
        self.pvalVec = None
        self.fitsBoloFirsts = None
        self.fitsBoloSeconds = None
        self.channel_errs = None
        self.preScaleVec = None
        self.minInd = None
        self.minDistType = None
        self.bestFitsBoloFirsts = None
        self.bestFitsBoloSeconds = None
        self.minPreScale = None
        self.minRadDist = None
        self.minRadDistList = None
        self.minPreScaleList = None
        self.minRadDistFits = None
        
    # calculates reduced chi^2 values for radiation structure library for one timestep
    def calc_fits(self, Etime, ErrorPool = False, PvalCutoff = None):
        
        if len(self.allRadDistsVec) == 0:
            self.load_raddists(TokamakName = self.tokamakAMode.tokamakName)
        
        if self.comparingTo == "Experiment":
            boloExpData = self.load_bolo_exp_timestep(EvalTime = Etime)
        elif self.comparingTo == "Simulation":
            boloExpData = self.load_radDist_as_exp()[Etime]
        
        self.chisqVec, self.pvalVec, self.fitsBoloFirsts, self.fitsBoloSeconds,\
            self.channel_errs, self.preScaleVec =\
            self.calc_pvals_JET(self.allRadDistsVec, BoloExpData = boloExpData)
        
        # take minimum pval RadDist, which is the closest fit
        if min(self.pvalVec) == 1.0: # covers case where Pvals bottom out (no pval better than 3). Not Ideal.
            self.minInd = np.argmin(self.chisqVec)
        else:
            self.minInd = np.argmin(self.pvalVec)
        self.minDistType = self.allRadDistsVec[self.minInd].distType
        self.bestFitsBoloFirsts = self.fitsBoloFirsts[self.minInd]
        self.bestFitsBoloSeconds = self.fitsBoloSeconds[self.minInd]
        self.minPreScale = self.preScaleVec[self.minInd]
        self.minchisq = self.chisqVec[self.minInd]
        print("best fit chi2 at this timestep is " + str(self.minchisq))
        self.minpval = self.pvalVec[self.minInd]
        self.minRadDist = self.allRadDistsVec[self.minInd]
        
        # Not yet set up for SPARC
        
        if ErrorPool:
            self.errorDists=[]
            self.errorFitsFirsts=[]
            self.errorFitsSeconds=[]
            self.errorPrescales=[]
            poolsize = 0
            for distNum in range(len(self.allRadDistsVec)):
                if self.pvalVec[distNum] <= PvalCutoff:
                    poolsize = poolsize + 1
                    self.errorDists.append(self.allRadDistsVec[distNum])
                    self.errorFitsFirsts.append(self.fitsBoloFirsts[distNum])
                    self.errorFitsSeconds.append(self.fitsBoloSeconds[distNum])
                    self.errorPrescales.append(self.preScaleVec[distNum])
            
            if not self.errorDists: # covers case where even best fit radDist is not within Pval cutoff.
                # Adds just best fit to error pool
                self.errorDists.append(self.minRadDist)
                self.errorFitsFirsts.append(self.bestFitsBoloFirsts)
                self.errorFitsSeconds.append(self.bestFitsBoloSeconds)
                self.errorPrescales.append(self.minPreScale)
            print(" pool size is " + str(poolsize))