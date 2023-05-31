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

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
            
            # removes any negative numbers from experimental data for prescaling.
            # negative channels should really be removed in fitting also; on JET, channel 16
            # removed in fitting function
            bolo_exp_for_sum = copy(bolo_exp)
            for indx1 in range(len(bolo_exp_for_sum)):
                for indx2 in range(len(bolo_exp_for_sum[indx1])):
                    if bolo_exp_for_sum[indx1][indx2] < 0:
                        bolo_exp_for_sum[indx1][indx2] = 0
            
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
            
    def gaussian(self, Phi, Sigma, Mu, Amplitude, OffsetVert):
            coeff = 1.0 / (Sigma * math.sqrt(2.0 * math.pi))
            exponential = np.exp(-((Phi - Mu) / Sigma)**2 / 2.0)
            return (Amplitude * coeff * exponential) + OffsetVert
    
    def gaussian_no_coeff(self, Phi, Sigma, Mu, Amplitude, OffsetVert):
            coeff = 1.0 / (Sigma * math.sqrt(2.0 * math.pi))
            return self.gaussian(Phi=Phi, Sigma=Sigma, Mu=Mu,
                                 Amplitude=Amplitude, OffsetVert=OffsetVert) / coeff
    
    # Returns an asymmetric gaussian. Implemented to handle arrays, since the curve_fit
    # function seems to need that
    def asymmetric_gaussian_arr(self, Phi, SigmaLeft, SigmaRight, Amplitude, Mu=None):
        if hasattr(Mu, "__len__"):
            mu=Mu
        elif Mu==None:
            mu = self.tokamakAMode.injectionPhiTor
        else:
            mu=Mu
        
        if hasattr(Phi, "__len__"): # check if Phi is array-like
            yval = []
            #for i in range(len(Phi)):
            for phi in Phi:
                #phi = Phi[i]
                if phi < mu:
                    yval.append(self.gaussian_no_coeff(Phi=phi, Sigma=SigmaLeft, Mu=mu,\
                            Amplitude=Amplitude, OffsetVert=0.0))
                else:
                    yval.append(self.gaussian_no_coeff(Phi=phi, Sigma=SigmaRight, Mu=mu,\
                            Amplitude=Amplitude, OffsetVert=0.0))
        else:
            if Phi < mu:
                yval=self.gaussian_no_coeff(Phi=Phi, Sigma=SigmaLeft, Mu=mu,\
                        Amplitude=Amplitude, OffsetVert=0.0)
            else:
                yval=self.gaussian_no_coeff(Phi=Phi, Sigma=SigmaRight, Mu=mu,\
                        Amplitude=Amplitude, OffsetVert=0.0)   
        
        return yval
            
    def calc_rad_error(self, PvalCutoff, MovePeak):
        # finds error bar ranges of rad power and tpf for a single timestep
        
        if not self.errorDists:
            raise Exception("Must run calc_fits with ErrorPool=True before calc_rad_error")
        
        errorRadPowers = []
        errorTpfs = []
        for distNum in range(len(self.errorDists)):
            try:
                errorRadPower, errorTpf, errorToroidalFitCoeffs =\
                    self.calc_rad_power(RadDist=self.errorDists[distNum], PreScale=self.errorPrescales[distNum],\
                        FitsFirsts=self.errorFitsFirsts[distNum], FitsSeconds=self.errorFitsSeconds[distNum],\
                        MovePeak=MovePeak)
                errorRadPowers.append(errorRadPower)
                errorTpfs.append(errorTpf)
            except:
                print("The toroidal distribution fitting of \
                      a helical distribution in pool of reasonable fits \
                      was unable to converge at this time")
        
        self.lowerBoundPower = min(errorRadPowers)
        self.upperBoundPower = max(errorRadPowers)
        self.lowerBoundTpf = min(errorTpfs)
        self.upperBoundTpf = max(errorTpfs)
            
        self.lowerBoundRadPowers.append(self.lowerBoundPower)
        self.upperBoundRadPowers.append(self.upperBoundPower)
        self.lowerBoundTpfs.append(self.lowerBoundTpf)
        self.upperBoundTpfs.append(self.upperBoundTpf)
            
        print("Lower Bound Rad Power is " + str(self.lowerBoundPower))
        print("Upper Bound Rad Power is " + str(self.upperBoundPower))
            
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
    
    # Output data from Emis3D as an excel file
    def output_to_excel(self):
        
        import pandas as pd
        
        # choose which variables to output to file, and what names they get in excel file
        vars_to_print = ["radPowerTimes", "minchisqList", "lowerBoundRadPowers", "radPowers",\
                         "upperBoundRadPowers", "lowerBoundTpfs", "tpfs", "upperBoundTpfs"]
        excel_names = ["Time (s)", "Best Chisq", "Prad Lower (GW)", "Prad Best (GW)",\
                       "Prad Upper (GW)", "TPF Lower", "TPF Best", "TPF Upper"]
        
        # copy all attributes of Emis3D instance as dictionary
        varsDict = copy(vars(self))
                
        # make new dictionary with only desired variables, renamed and ordered as excel_names
        outputDict = {}
        for name in excel_names:
            values = varsDict[vars_to_print[excel_names.index(name)]]
            # also convert radiated powers from W to GW
            if name in ["Prad Lower (GW)", "Prad Best (GW)", "Prad Upper (GW)"]:
                values = [x * 1e-9 for x in values]
            outputDict[name] = values
        
        # convert to pandas dataframe
        outputDf = pd.DataFrame.from_dict(outputDict)
        
        # output as excel
        writer = pd.ExcelWriter(join(self.data_output_directory, str(self.shotnumber) + ".xlsx"))
        outputDf.to_excel(writer, sheet_name=str(self.shotnumber), startcol=0)
            
        # add any additional tokamak-specific variables to dictionary
        try:
            outputExtrasDict = self.tok_specific_output_data()
            outputExtrasDf = pd.DataFrame.from_dict(outputExtrasDict)
            outputExtrasDf.to_excel(writer, sheet_name=str(self.shotnumber),\
                                        startcol=len(excel_names) + 2)
        except:
            pass
        
        # close excel writer
        writer.close()
    
    def plot_radDist_array(self, PlotData, Levels,\
                                  Lengthx, Lengthy, Etime, ColorBarLabel, PlotDistType):
        
        # Pull only RadDists of given type to be plotted
        plotRVecs = []
        plotZVecs = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                thisRadDist = self.allRadDistsVec[distNum]
                plotRVecs.append(thisRadDist.startR)
                plotZVecs.append(thisRadDist.startZ)
                
        # Create Figure
        fig = plt.figure(figsize=(Lengthx, Lengthy))
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.set_xlabel('$R$ (m)')
        ax.set_ylabel('$Z$ (m)')
        
        # Create color contour of fits
        tcf = ax.tricontourf(plotRVecs, plotZVecs, PlotData,\
            levels=Levels, cmap='gist_stern')
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
        fig.colorbar(tcf, ax=ax, label=ColorBarLabel, shrink=0.5, format='%d')
        
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
    
    def plot_fits_array(self, Lengthx=3, Lengthy=4, Etime = 50.89, PlotDistType = "Helical"):
        
        if not self.pvalVec:
            raise Exception("Must calculate p values before plotting fits")
            
        # Pull only RadDists of given type to be plotted
        plotChisq = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                plotChisq.append(self.chisqVec[distNum])
        
        levels = np.linspace(0, 40., num=20)
        
        fig = self.plot_radDist_array(\
            PlotData=plotChisq, Levels=levels, Lengthx=Lengthx, Lengthy=Lengthy, Etime=Etime,\
            ColorBarLabel='$\chi^2_r$', PlotDistType=PlotDistType)
        
        return fig
        
    
    def plot_powers_array(self, Lengthx=3, Lengthy=4, Etime = 50.89, PlotDistType = "Helical",\
                          MovePeak=False):
        
        if not self.pvalVec:
            raise Exception("Must calculate p values before plotting powers")
    
        # Pull only RadDists of given type to be plotted
        plotDistsRadPowers = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                thisRadDist = self.allRadDistsVec[distNum]
                
                radPower = self.calc_rad_power(RadDist=thisRadDist, PreScale=self.preScaleVec[distNum],\
                    FitsFirsts=self.fitsBoloFirsts[distNum], FitsSeconds=self.fitsBoloSeconds[distNum],\
                    MovePeak=MovePeak)[0]

                plotDistsRadPowers.append(radPower)
                
        bestFitRadDist = self.allRadDistsVec[self.minInd]
        bestFitRadPower = self.calc_rad_power(RadDist=bestFitRadDist, PreScale=self.preScaleVec[self.minInd],\
                    FitsFirsts=self.fitsBoloFirsts[self.minInd], FitsSeconds=self.fitsBoloSeconds[self.minInd],\
                    MovePeak=MovePeak)[0]
        print("Best Fit Rad Power is " + str(bestFitRadPower / 1e9) + "GW")
        
        #This step removes very high outliers that screw up the plot
        #medPower = np.median(plotDistsRadPowers)
        
        for i in range(len(plotDistsRadPowers)):
            if plotDistsRadPowers[i] > (10.0 * bestFitRadPower):
                plotDistsRadPowers[i] = 0.0
        
        levels = np.linspace(0, max(plotDistsRadPowers)/1e9, num=20)
        
        fig = self.plot_radDist_array(\
            PlotData=np.array(plotDistsRadPowers)/1e9, Levels=levels,\
            Lengthx=Lengthx, Lengthy=Lengthy, Etime=Etime,\
            ColorBarLabel='$P_{rad}$ (GW)', PlotDistType = PlotDistType)
        
        return fig
    
    def plot_fits_channels(self, Etime = 50.89, AsBrightness = True):
        
        bolo_exp, synthArrayFirst, synthArraySecond = self.plot_fits_data_organization(Etime=Etime)
        
        fig, axs = plt.subplots(self.numTorLocs, 1, figsize=(3,2*self.numTorLocs))
        
        channel_errs = self.channel_errs
        
        for numTor in range(self.numTorLocs):
            ax = axs[numTor]
    
            #Remove error bars that are larger than measurement itself, just for plotting
            for errIndex in range(len(channel_errs[numTor])):
                try:
                    if abs(channel_errs[numTor][errIndex]) > abs(bolo_exp[numTor][errIndex]):
                        channel_errs[numTor][errIndex] = 0.0
                except:
                    pass
                    
            # convert to power per m^2
            if AsBrightness:
                for i in range(len(bolo_exp[numTor])):
                    bolo_exp[numTor][i] = bolo_exp[numTor][i] * 4.0 * math.pi / self.bolo_etendues[numTor][i]
                    synthArrayFirst[numTor][i] = synthArrayFirst[numTor][i] * 4.0 * math.pi / self.bolo_etendues[numTor][i]
                    # only works if there is a second puncture, otherwise length of list is wrong
                    try:
                        synthArraySecond[numTor][i] = synthArraySecond[numTor][i] * 4.0 * math.pi / self.bolo_etendues[numTor][i]
                    except:
                        pass
                

                # for MW / m^2
                ax.set_ylabel(r"Power Location " + str(numTor) + " $(MW/m^2)$")
                bolo_exp[numTor] = [x * 1e-6 for x in bolo_exp[numTor]]
                channel_errs[numTor] = [x * 1e-6 for x in channel_errs[numTor]]    
                synthArrayFirst[numTor] = [x * 1e-6 for x in synthArrayFirst[numTor]]
                try:
                    synthArraySecond[numTor] = [x * 1e-6 for x in synthArraySecond[numTor]]
                except:
                    pass
            
            else:
                # for watts
                multiplier = 1.0
                ax.set_ylabel(r"Power Location " + str(numTor) + " $(W)$")
                # for milliwatts
                #multiplier =1e3
                #ax.set_ylabel(r"Power Location " + str(numTor) + " $(mW)$")
                
                bolo_exp[numTor] = [x * multiplier for x in bolo_exp[numTor]]
                channel_errs[numTor] = [x * multiplier for x in channel_errs[numTor]]    
                synthArrayFirst[numTor] = [x * multiplier for x in synthArrayFirst[numTor]]    
                synthArraySecond[numTor] = [x * multiplier for x in synthArraySecond[numTor]] 
            
            channelNum = range(len(bolo_exp[numTor]))
            try:
                ax.errorbar(channelNum, bolo_exp[numTor], yerr=channel_errs[numTor], label='Experiment', color='blue')
            except:
                ax.plot(channelNum, bolo_exp[numTor], label='Experiment', color='blue')
            ax.plot(channelNum, synthArrayFirst[numTor], '-s', label='1st Puncture', color='gray')
            try:
                ax.plot(channelNum, synthArraySecond[numTor], '-x', label='2nd Puncture', color='gray')
                ax.plot(channelNum, ([synthArrayFirst[numTor][i] + synthArraySecond[numTor][i]\
                    for i in range(len(synthArrayFirst[numTor]))]),\
                    '-o', label='Full Synthetic', color='green')
            except:
                pass
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        plt.tight_layout()
        
        return fig
    
    def save_bolos_contour_plot(self, Times, Bolo_vals,\
            Title, SaveName, SaveFolder,\
            LowerBound=4, UpperBound=8.2, NumTicks=22):
        
        channel_list = np.array(range(len(Bolo_vals)))
        x, y = np.meshgrid(Times, channel_list)
        
        plot_vals = copy(Bolo_vals)
        for row in range(len(plot_vals)):
            for indx in range(len(plot_vals[row])):
                if plot_vals[row][indx] > 0:
                    plot_vals[row][indx] = np.log10(plot_vals[row][indx])
                else:
                    plot_vals[row][indx] = np.nan
        
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(x, y, plot_vals, levels = np.linspace(LowerBound, UpperBound, NumTicks), cmap="jet")
        colorbar = fig.colorbar(cp)
        colorbar.ax.set_ylabel("Log(Brightness [$W/m^2$])")
        ax.set_title(Title)
        ax.set_ylabel('Channel Number')
        ax.set_xlabel('Time [s]')
        savefile = join(SaveFolder, SaveName) + ".png"
        plt.savefig(savefile, format='png')
        
        plt.close(fig)
        
    def save_synth_contour_plot(self, ArrayNum, SaveName, SaveFolder,\
                                PreviousArrayNum=None, NumChannels=None, LowerBound=1e4):
        
        numTimes = len(self.radPowerTimes)
        if NumChannels==None:
            numChannels=0
            bolo_powers_1st = self.rearrange_powers_array(self.minRadDistList[0].boloCameras_powers)
            if hasattr(bolo_powers_1st[ArrayNum][0], "len"):
                for subCameraNum in range(len(bolo_powers_1st[ArrayNum])):
                    numChannels += len(bolo_powers_1st[ArrayNum][subCameraNum])
            else:
                numChannels += len(bolo_powers_1st[ArrayNum])
        else:
            numChannels=NumChannels
        
        etendues = self.bolo_etendues[ArrayNum]
        bolos = np.zeros([numTimes, numChannels])
        for timeIndx in range(numTimes):
            plotFitsFirsts = self.minFitsFirsts[timeIndx]
            plotFitsSeconds = self.minFitsSeconds[timeIndx]
            preScale = self.minPreScaleList[timeIndx]
            minRadDist = copy(self.minRadDistList[timeIndx])
            for channelIndx in range(numChannels):
                boloValue=0.0
                bolo_powers = self.rearrange_powers_array(self.minRadDistList[timeIndx].boloCameras_powers)
                bolo_powers_2nd = self.rearrange_powers_array(self.minRadDistList[timeIndx].boloCameras_powers_2nd)
                if hasattr(bolo_powers[ArrayNum][0], "len"):
                    for subCameraNum in range(len(bolo_powers[ArrayNum])):
                        boloValue = bolo_powers[ArrayNum][subCameraNum][channelIndx]\
                            * preScale * plotFitsFirsts[ArrayNum]
                        if hasattr(bolo_powers_2nd[ArrayNum][subCameraNum], "len")\
                            and PreviousArrayNum != None:
                            boloValue += bolo_powers_2nd[ArrayNum][subCameraNum][channelIndx]\
                                * preScale * plotFitsFirsts[PreviousArrayNum] * plotFitsSeconds[ArrayNum]
                else:
                    boloValue = bolo_powers[ArrayNum][channelIndx]\
                        * preScale * plotFitsFirsts[ArrayNum]
                    if hasattr(minRadDist.boloCameras_powers_2nd[ArrayNum], "len")\
                        and PreviousArrayNum != None:
                        boloValue += bolo_powers_2nd[ArrayNum][channelIndx]\
                            * preScale * plotFitsFirsts[PreviousArrayNum] * plotFitsSeconds[ArrayNum]
                boloValue = boloValue * 4.0 * math.pi / etendues[channelIndx]
                bolos[timeIndx][channelIndx] = boloValue
        
        # sets lower bound to plot values
        for i in range(numTimes):
            for j in range(numChannels):
                if bolos[i][j] < LowerBound:
                    bolos[i][j] = LowerBound
                
        bolos = bolos.T
        
        self.save_bolos_contour_plot(Times = self.radPowerTimes, Bolo_vals = bolos,\
            Title = "Array " + str(ArrayNum) + " Best Fit Brightnesses\nDischarge " + str(self.shotnumber),\
            SaveName = SaveName,\
            SaveFolder = SaveFolder)
        
    def save_exp_contour_plot(self, ArrayNum, Title, SaveName, SaveFolder,\
        StartTime=None, EndTime=None, EndChannel=None, DeleteChannels=None):
        
        if StartTime==None and EndTime==None:
            startTime = self.radPowerTimes[0]
            endTime = self.radPowerTimes[-1]
        else:
            startTime=StartTime
            endTime=EndTime
        
        if EndChannel != None:
            expData = self.load_bolo_exp_timerange(StartTime=startTime, EndTime=endTime,\
                AsBrightnesses=True)[ArrayNum][:EndChannel]
        else:
            expData = self.load_bolo_exp_timerange(StartTime=startTime, EndTime=endTime,\
            AsBrightnesses=True)[ArrayNum]
            
        expTimebase = self.load_bolo_timebase_range(StartTime=startTime, EndTime=endTime)[ArrayNum]
        
        if DeleteChannels != None:
            for indx in range(len(DeleteChannels)):
                channel = DeleteChannels[indx]
                expData[channel] = np.array([np.nan] * len(expData[channel]))
        
        self.save_bolos_contour_plot(Times = expTimebase, Bolo_vals = expData,\
            Title = Title, SaveName = SaveName, SaveFolder = SaveFolder)
        
    def save_sim_contour_plot(self, ArrayNum, Title, SaveName, SaveFolder,\
        EndChannel=None, DeleteChannels=None):
        
        simData = copy(self.bolo_sim)
        
        for timeIndx in range(len(simData)):
            simData[timeIndx] = self.rearrange_powers_array(Powers=simData[timeIndx])
        
        if EndChannel != None:
            for timeIndx in range(len(simData)):
                simData[timeIndx] = simData[timeIndx][ArrayNum][:EndChannel]
        else:
            for timeIndx in range(len(simData)):
                simData[timeIndx] = simData[timeIndx][ArrayNum]
        
        for timeIndx in range(len(simData)):
            for channelIndx in range(len(simData[timeIndx])):
                boloValue = simData[timeIndx][channelIndx]
                boloValue = boloValue * 4.0 * math.pi / self.bolo_etendues[ArrayNum][channelIndx]
                simData[timeIndx][channelIndx] = boloValue
        
        simData = np.array(simData).T
        
        simTimebase = self.radPowerTimes
        
        if DeleteChannels != None:
            for indx in range(len(DeleteChannels)):
                channel = DeleteChannels[indx]
                simData[channel] = np.array([np.nan] * len(simData[channel]))
        
        self.save_bolos_contour_plot(Times = simTimebase, Bolo_vals = simData,\
            Title = Title, SaveName = SaveName, SaveFolder = SaveFolder)
        
    def make_crossSec_movie(self, Phi=0.0):
        
        self.load_tokamak(Mode="Build")
        numFillZeros = len(str(len(self.minRadDistList)))
        for radDistNum in range(len(self.minRadDistList)):
            savefile = join(self.videos_output_directory, "crossSecImg") +\
                        str(radDistNum).zfill(numFillZeros) + ".png"
            radDist = copy(self.minRadDistList[radDistNum])
            radDist.set_tokamak(self.tokamakBMode)
            radDist.make_build_mode()
            
            crossSecPlot = radDist.plot_crossSec(Phi=Phi)

            crossSecPlot.savefig(savefile, format='png')
            plt.close(crossSecPlot)
        
        """
        import cv2

        video_name = 'Animations/crossSecs.avi'

        images = [img for img in listdir(image_folder) if img.endswith(".png") and img.startswith('crossSecImg')]
        images.sort()
        frame = cv2.imread(join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(join(image_folder, image)))
            
        for radDistNum in range(len(self.minRadDistList)):
            savefile = join(image_folder, "crossSecImg") +\
                        str(radDistNum).zfill(numFillZeros) + ".png"
            remove(savefile)

        video.release()
        """
        #return video