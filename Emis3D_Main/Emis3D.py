#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:55:26 2023

@author: br0148
"""

from os import listdir
from os.path import dirname, realpath, isfile, join
        
FILE_PATH = dirname(realpath(__file__))
EMIS3D_PARENT_DIRECTORY = dirname(dirname(FILE_PATH))
EMIS3D_UNIVERSAL_MAIN_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY,\
    "Emis3D_Universal", "Emis3D_Main")
EMIS3D_INPUTS_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "Emis3D_JET", "Emis3D_Inputs")

import numpy as np
import matplotlib.pyplot as plt
from Util import RedChi2_To_Pvalue
from scipy.optimize import minimize
from copy import copy

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
        
    def load_raddists(self, TokamakName):
        
        self.allRadDistsVec = []
        
        # returns all files in RadDist_Saves folder
        onlyfiles = [f for f in listdir(self.raddist_saves_directory)\
                     if isfile(join(self.raddist_saves_directory, f))]
        
        for distIndx in range(len(onlyfiles)):
            loadFileName = join(self.raddist_saves_directory,onlyfiles[distIndx])
            radDist = self.load_single_raddist(LoadFileName = loadFileName)
            
            self.allRadDistsVec.append(radDist)
            
    def calc_pvals(self, RadDistVec, BoloExpData, PowerUpperBound = 6000.0):
        
        print("Calculating P-Values")
        
        numChannels = np.sum([len(x) for x in BoloExpData])
        
        bolo_exp = self.rearrange_powers_array(Powers = BoloExpData)
        
        chisqlist = []
        pValList = []
        fitsBoloFirsts = []
        fitsBoloSeconds = []
        preScales = []
        channel_errs = []
        
        # this is for different errors at each toroidal location
        for i in range(self.numTorLocs):
            expMax = np.max(bolo_exp[i])
            channel_errs.append(np.array([0.1 * expMax] * len(bolo_exp[i])))

        for radDist in RadDistVec:
            synth_powers_p1 = self.rearrange_powers_array(copy(radDist.boloCameras_powers))
            synth_powers_p2 = self.rearrange_powers_array(copy(radDist.boloCameras_powers_2nd))
            
            # uniformly pre-scales synthetic powers to same order of magnitude as experimental
            # data, to put in range of fitting algorithm
            preScaleFactorNum = np.sum([np.sum(bolo_exp[indx]) for indx in range(len(bolo_exp))])
            preScaleFactorDenom = np.sum([np.sum(synth_powers_p1[indx]) for indx in range(len(synth_powers_p1))])
            preScaleFactor = preScaleFactorNum / preScaleFactorDenom
                
            for i in range(self.numTorLocs):
                synth_powers_p1[i] = np.array(synth_powers_p1[i]) * preScaleFactor
                synth_powers_p2[i] = np.array(synth_powers_p2[i]) * preScaleFactor
            
            p0Firsts, p0Seconds, boundsFirsts, boundsSeconds, fittingFunc =\
                self.fitting_func_setup(Bolo_exp=bolo_exp, Synth_powers_p1=synth_powers_p1,\
                    Synth_powers_p2=synth_powers_p2, Channel_errs=channel_errs, PowerUpperBound=PowerUpperBound)

            res = minimize(fittingFunc, p0Firsts + p0Seconds, bounds=boundsFirsts+boundsSeconds)
            res.fun
            fitsBoloFirsts.append(res.x[:self.numTorLocs])
            fitsBoloSeconds.append(res.x[self.numTorLocs:])
            preScales.append(preScaleFactor)
            
            if radDist.distType == "Helical":
                degree_of_freedom = numChannels - (len(p0Firsts) + len(p0Seconds) + 1)
            else:
                degree_of_freedom = numChannels - (len(p0Firsts) + 1)
            chisq = (res.fun)/degree_of_freedom
            chisqlist.append(chisq)
            pValList.append(RedChi2_To_Pvalue(chisq, degree_of_freedom))
        
        return chisqlist, pValList, fitsBoloFirsts, fitsBoloSeconds, channel_errs, preScales
            
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
            self.calc_pvals(self.allRadDistsVec, BoloExpData = boloExpData)
        
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
            
    def calc_tot_rad(self, TimePerStep, NumTimes = 11,\
                     ErrorPool = False, PvalCutoff = 0.9545, MovePeak=False):
        # finds total radiated energy for entire shot. Stores timestep-by-timestep best fit
        # information in the process, and error bound information if ErrorPool=True
        
        self.radPowers = []
        self.tpfs = []
        self.minRadDistList = []
        self.minPreScaleList = []
        self.minFitsFirsts = []
        self.minFitsSeconds = []
        self.minchisqList = []
        self.minpvalList = []
        self.minTorFitCoeffs = []
        
        wrad = 0.0
        upperwrad = 0.0
        lowerwrad = 0.0
        self.accumWrads = []
        self.upperAccumWrads = []
        self.lowerAccumWrads = []
        
        self.errorDists = []
        self.lowerBoundRadPowers = []
        self.upperBoundRadPowers = []
        self.lowerBoundTpfs = []
        self.upperBoundTpfs = []
        
        for timeIndex in range(NumTimes):
            print(" ")
            print("calculating time " + str(timeIndex) + "/" + str(NumTimes))
            
            self.calc_fits(Etime=self.radPowerTimes[timeIndex], ErrorPool=ErrorPool, PvalCutoff=PvalCutoff)
                
            print("Best Fit RadDist Type = " + self.minDistType)
            self.minRadDistList.append(self.minRadDist)
            self.minPreScaleList.append(self.minPreScale)
            self.minFitsFirsts.append(self.bestFitsBoloFirsts)
            self.minFitsSeconds.append(self.bestFitsBoloSeconds)
            self.minchisqList.append(self.minchisq)
            self.minpvalList.append(self.minpval)
                
            radPower, tpf, torFitCoeffs =\
                self.calc_rad_power(RadDist=self.minRadDist, PreScale=self.minPreScale,\
                        FitsFirsts=self.bestFitsBoloFirsts, FitsSeconds=self.bestFitsBoloSeconds,\
                        MovePeak=MovePeak)
                
            self.radPowers.append(radPower)
            self.tpfs.append(tpf)
            self.minTorFitCoeffs.append(torFitCoeffs)
                
            print("Rad Power in GW is " + str(radPower * 1e-9))
            print("TPF is " + str(tpf))
            
            if ErrorPool:
                self.calc_rad_error(PvalCutoff=PvalCutoff, MovePeak=MovePeak)
            
            wrad = wrad + (radPower * TimePerStep)
            upperwrad = upperwrad + (self.upperBoundPower * TimePerStep)
            lowerwrad = lowerwrad + (self.lowerBoundPower*TimePerStep)
            self.accumWrads.append(wrad)
            self.upperAccumWrads.append(upperwrad)
            self.lowerAccumWrads.append(lowerwrad)
            
        self.totWrad = self.accumWrads[-1]
        self.upperTotWrad = self.upperAccumWrads[-1]
        self.lowerTotWrad = self.lowerAccumWrads[-1]
        
    def plot_fits_array(self, Lengthx=3, Lengthy=4, Etime = 50.89, PlotDistType = "Helical"):
        
        if not self.pvalVec:
            raise Exception("Must calculate p values before plotting fits")
            
        # Pull only RadDists of given type to be plotted
        plotDistsVec = []
        plotRVecs = []
        plotZVecs = []
        plotChisq = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                thisRadDist = self.allRadDistsVec[distNum]
                plotDistsVec.append(thisRadDist)
                plotRVecs.append(thisRadDist.startR)
                plotZVecs.append(thisRadDist.startZ)
                plotChisq.append(self.chisqVec[distNum])
            
        # Create Figure
        fig = plt.figure(figsize=(Lengthx, Lengthy))
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.set_xlabel('$R$ (m)')
        ax.set_ylabel('$Z$ (m)')
        
        # Create color contour of fits
        levels = np.linspace(0, 40., num=20)
        tcf = ax.tricontourf(plotRVecs, plotZVecs, plotChisq,\
            levels=levels, cmap='gist_stern')
        ax.plot(plotRVecs, plotZVecs, 'xk')
        
        # Add white x and circle around best fit
        if self.minDistType == PlotDistType:
            bestRadDist = self.allRadDistsVec[self.minInd]
            r = bestRadDist.startR
            z = bestRadDist.startZ
            ax.plot(r, z, 'o', markeredgecolor='white', fillstyle='none', 
                    markeredgewidth=3, markersize=40)
            ax.plot(r, z, 'x', markeredgecolor='white')
            
        # Set plot dimensions
        ax.set_xlim([self.tokamakAMode.majorRadius - (1.1*self.tokamakAMode.minorRadius),\
                     self.tokamakAMode.majorRadius + (1.1*self.tokamakAMode.minorRadius)])
        ax.set_ylim([(-2.4*self.tokamakAMode.minorRadius),\
                     (2.4*self.tokamakAMode.minorRadius)])
    
        # Set plot title
        ax.set_title(str(PlotDistType) + "s")
        
        # Set colorbar
        fig.colorbar(tcf, ax=ax, label='$\chi^2_r$', shrink=0.5, format='%d')
        
        # Plot first wall curve
        r = self.tokamakAMode.wallcurve.vertices[:,0]
        z = self.tokamakAMode.wallcurve.vertices[:,1]
        ax.plot(r,z, 'orange')
        
        # Plot Q=1.1,2, and 3 surfaces
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval=1.1)
        ax.plot(r,z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval=2.0)
        ax.plot(r,z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval=3.0)
        ax.plot(r,z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval="Separatrix")
        ax.plot(r,z, "cyan")
        
        return fig
    
    def plot_powers_array(self, Lengthx=3, Lengthy=4, Etime = 50.89, PlotDistType = "Helical",\
                          MovePeak=False):
        
        if not self.pvalVec:
            raise Exception("Must calculate p values before plotting powers")
    
        # Pull only RadDists of given type to be plotted
        plotDistsVec = []
        plotRVecs = []
        plotZVecs = []
        plotDistsRadPowers = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                thisRadDist = self.allRadDistsVec[distNum]
                plotDistsVec.append(thisRadDist)
                plotRVecs.append(thisRadDist.startR)
                plotZVecs.append(thisRadDist.startZ)
                
                radPower = self.calc_rad_power(RadDist=thisRadDist, PreScale=self.preScaleVec[distNum],\
                    FitsFirsts=self.fitsBoloFirsts[distNum], FitsSeconds=self.fitsBoloSeconds[distNum],\
                    MovePeak=MovePeak)[0]

                plotDistsRadPowers.append(radPower)
                
        bestFitRadDist = self.allRadDistsVec[self.minInd]
        bestFitRadPower = self.calc_rad_power(RadDist=bestFitRadDist, PreScale=self.preScaleVec[self.minInd],\
                    FitsFirsts=self.fitsBoloFirsts[self.minInd], FitsSeconds=self.fitsBoloSeconds[self.minInd],\
                    MovePeak=MovePeak)[0]
        print("Best Fit Rad Power is " + str(bestFitRadPower / 1e9) + "GW")
        
        # Create Figure
        fig = plt.figure(figsize=(Lengthx, Lengthy))
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.set_xlabel('$R$ (m)')
        ax.set_ylabel('$Z$ (m)')
        
        #This step removes very high outliers that screw up the plot
        #medPower = np.median(plotDistsRadPowers)
        
        for i in range(len(plotDistsRadPowers)):
            if plotDistsRadPowers[i] > (10.0 * bestFitRadPower):
                plotDistsRadPowers[i] = 0.0
        
        # Create color contour of powers
        levels = np.linspace(0, max(plotDistsRadPowers)/1e9, num=20)
        tcf = ax.tricontourf(plotRVecs, plotZVecs, np.array(plotDistsRadPowers)/1e9,\
            levels=levels, cmap='gist_stern')
        ax.plot(plotRVecs, plotZVecs, 'xk')
        
        # Add white x and circle around best fit
        if self.minDistType == PlotDistType:
            bestRadDist = self.allRadDistsVec[self.minInd]            
            
            r = bestRadDist.startR
            z = bestRadDist.startZ
            ax.plot(r, z, 'o', markeredgecolor='white', fillstyle='none', 
                    markeredgewidth=3, markersize=40)
            ax.plot(r, z, 'x', markeredgecolor='white')
        
        # Set plot dimensions
        ax.set_xlim([self.tokamakAMode.majorRadius - (1.1*self.tokamakAMode.minorRadius),\
                     self.tokamakAMode.majorRadius + (1.1*self.tokamakAMode.minorRadius)])
        ax.set_ylim([(-2.4*self.tokamakAMode.minorRadius),\
                     (2.4*self.tokamakAMode.minorRadius)])
    
        # Set plot title
        ax.set_title(str(PlotDistType) + "s")
        
        # Set colorbar
        fig.colorbar(tcf, ax=ax, label='$P_{rad}$ (GW)', shrink=0.5, format='%.2f')
        
        # For plotting first wall curve
        r = self.tokamakAMode.wallcurve.vertices[:,0]
        z = self.tokamakAMode.wallcurve.vertices[:,1]
        ax.plot(r,z, 'orange')
        
        # For plotting q surfaces
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval=1.1)
        ax.plot(r,z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval=2.0)
        ax.plot(r,z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval=3.0)
        ax.plot(r,z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(Shotnumber=self.shotnumber, EvalTime=Etime-5e-4, Qval="Separatrix")
        ax.plot(r,z, "cyan")
        
        return fig