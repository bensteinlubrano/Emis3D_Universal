#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:55:26 2023

@author: br0148
"""

import sys
from os import listdir, remove
from os.path import dirname, isdir, isfile, join, realpath

FILE_PATH = dirname(realpath(__file__))
EMIS3D_PARENT_DIRECTORY = dirname(dirname(FILE_PATH))
EMIS3D_UNIVERSAL_MAIN_DIRECTORY = join(
    EMIS3D_PARENT_DIRECTORY, "Emis3D_Universal", "Emis3D_Main"
)
EMIS3D_INPUTS_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "Emis3D_JET", "Emis3D_Inputs")

import math
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit, minimize
from Util import RedChi2_To_Pvalue


class Emis3D(object):

    def __init__(self, TorSegmented=False):

        self.torSegmented = TorSegmented
        self.allRadDistsVec = []
        self.tokamakAMode = None  # Always loaded
        self.tokamakBMode = None  # Not loaded unless necessary; time consuming
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
        self.minFitsFirsts = None
        self.minFitsSeconds = None
        self.extrasPowers = None

    def load_raddists(self, TokamakName):

        self.allRadDistsVec = []

        if TokamakName == "JET":
            # returns all files in self.raddist_directories_to_load
            onlyfiles = []
            for directory in self.raddist_directories_to_load:
                raddistfiles = [
                    join(directory, f)
                    for f in listdir(directory)
                    if isfile(join(directory, f))
                ]
                for f in raddistfiles:
                    onlyfiles.append(f)

        else:
            # returns all files in RadDist_Saves folder
            onlyfiles = [
                join(self.raddist_saves_directory, f)
                for f in listdir(self.raddist_saves_directory)
                if isfile(join(self.raddist_saves_directory, f))
            ]

        for distIndx in range(len(onlyfiles)):
            loadFileName = onlyfiles[distIndx]
            radDist = self.load_single_raddist(LoadFileName=loadFileName)

            self.allRadDistsVec.append(radDist)

    def calc_pvals(self, RadDistVec, BoloExpData, PowerUpperBound=6000.0):

        print("Calculating P-Values")

        numChannels = np.sum([len(x) for x in BoloExpData])

        bolo_exp = self.rearrange_powers_array(Powers=BoloExpData)

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
            synth_powers_p1 = self.rearrange_powers_array(
                Powers=copy(radDist.boloCameras_powers)
            )
            synth_powers_p2 = self.rearrange_powers_array(
                Powers=copy(radDist.boloCameras_powers_2nd)
            )

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
            preScaleFactorNum = np.sum(
                [np.sum(bolo_exp[indx]) for indx in range(len(bolo_exp))]
            )
            preScaleFactorDenom = np.sum(
                [np.sum(synth_powers_p1[indx]) for indx in range(len(synth_powers_p1))]
            )
            preScaleFactor = preScaleFactorNum / preScaleFactorDenom

            for i in range(self.numTorLocs):
                synth_powers_p1[i] = np.array(synth_powers_p1[i]) * preScaleFactor
                if len(synth_powers_p2) > 0:  # check if there's a second puncture
                    synth_powers_p2[i] = np.array(synth_powers_p2[i]) * preScaleFactor

            p0Firsts, p0Seconds, boundsFirsts, boundsSeconds, fittingFunc = (
                self.fitting_func_setup(
                    Bolo_exp=bolo_exp,
                    Synth_powers_p1=synth_powers_p1,
                    Synth_powers_p2=synth_powers_p2,
                    Channel_errs=channel_errs,
                    PowerUpperBound=PowerUpperBound,
                )
            )

            res = minimize(
                fittingFunc, p0Firsts + p0Seconds, bounds=boundsFirsts + boundsSeconds
            )
            res.fun
            fitsBoloFirsts.append(res.x[: self.numTorLocs])
            fitsBoloSeconds.append(res.x[self.numTorLocs :])
            preScales.append(preScaleFactor)

            if radDist.distType == "Helical":
                degree_of_freedom = numChannels - (len(p0Firsts) + len(p0Seconds) + 1)
            else:
                degree_of_freedom = numChannels - (len(p0Firsts) + 1)
            chisq = (res.fun) / degree_of_freedom
            chisqlist.append(chisq)
            pValList.append(RedChi2_To_Pvalue(chisq, degree_of_freedom))

        return (
            chisqlist,
            pValList,
            fitsBoloFirsts,
            fitsBoloSeconds,
            channel_errs,
            preScales,
        )

    def calc_pvals_segmented(self, RadDistVec, BoloExpData, PowerUpperBound=6000.0):
        # only set up for SPARC, and the July 2023 Bolo configuration

        print("Calculating P-Values")
        numChannels = (
            self.tokamakAMode.numTorLocs
            * self.tokamakAMode.cameras_per_tor
            * self.tokamakAMode.channelsPerCamera
        )

        exp_powers_unsegmented = copy(BoloExpData)
        maxExpPower = np.max(exp_powers_unsegmented)
        minPowerCutoff = 0.0  # 0.01*maxExpPower

        chisqlist = []
        pValList = []
        fitsBoloFirsts = []
        fitsBoloSeconds = []
        preScales = []
        channel_errs = []

        for radDist in RadDistVec:
            if radDist.distType == "Helical" or radDist.distType == "ElongatedHelical":

                synth_powers_unsegmented = copy(radDist.boloCameras_powers)
                synth_powers_segmented = copy(radDist.bolo_powers_segmented)
                synth_powers_segmented_2nd = copy(radDist.bolo_powers_segmented_2nd)
                num_segments = len(synth_powers_segmented[0][0])

                # Creates an array of angles which correspond to the center of each segment.
                # first half are positive, second half are negative: goes from 0 to pi then
                # -pi to 0. To match how the powers array indexing works. Then goes around a
                # second time for second punctures.
                halfSegmentPhiWidth = np.pi / num_segments
                half_num_segments = int(num_segments / 2)
                segmentCenterPhis1 = np.linspace(
                    halfSegmentPhiWidth, np.pi - halfSegmentPhiWidth, half_num_segments
                )
                segmentCenterPhis2 = np.linspace(
                    -np.pi + halfSegmentPhiWidth,
                    -halfSegmentPhiWidth,
                    half_num_segments,
                )
                segmentCenterPhis3 = np.linspace(
                    -(2.0 * np.pi) + halfSegmentPhiWidth,
                    -np.pi - halfSegmentPhiWidth,
                    half_num_segments,
                )
                segmentCenterPhis4 = np.linspace(
                    np.pi + halfSegmentPhiWidth,
                    (2.0 * np.pi) - halfSegmentPhiWidth,
                    half_num_segments,
                )
                segmentCenterPhis = np.concatenate(
                    (
                        segmentCenterPhis1,
                        segmentCenterPhis2,
                        segmentCenterPhis3,
                        segmentCenterPhis4,
                    )
                )

                # uniformly pre-scales synthetic powers to same order of magnitude as experimental
                # data, to put in range of fitting algorithm
                preScaleFactorNum = np.sum(exp_powers_unsegmented)
                preScaleFactorDenom = np.sum(synth_powers_unsegmented)
                preScaleFactor = preScaleFactorNum / preScaleFactorDenom

                numIgnoredChannels = 0
                # this loop for counting the number of ignored channels for the purposes of the degree of freedom
                # and for uniformly rescaling the synthetic powers to start closer to the experimental values
                # currently, no channels are ignored
                for CameraIndx in range(len(exp_powers_unsegmented)):
                    for channelIndx in range(len(exp_powers_unsegmented[CameraIndx])):
                        channelExpPower = exp_powers_unsegmented[CameraIndx][
                            channelIndx
                        ]
                        if channelExpPower <= minPowerCutoff:
                            numIgnoredChannels += 1
                        for channelSegmentIndx in range(
                            len(synth_powers_segmented[CameraIndx][channelIndx])
                        ):
                            segmentSynthPower = synth_powers_segmented[CameraIndx][
                                channelIndx
                            ][channelSegmentIndx]
                            synth_powers_segmented[CameraIndx][channelIndx][
                                channelSegmentIndx
                            ] = (segmentSynthPower * preScaleFactor)

                p0Firsts, p0Seconds, boundsFirsts, boundsSeconds, fittingFunc = (
                    self.fitting_func_setup_segmented_helical(
                        Bolo_exp_unsegmented=exp_powers_unsegmented,
                        Synth_powers_segmented=synth_powers_segmented,
                        PowerUpperBound=PowerUpperBound,
                        MinPowerCutoff=minPowerCutoff,
                        SegmentCenterPhis=segmentCenterPhis,
                        MaxExpPower=maxExpPower,
                        NumPuncs=2,
                        Synth_powers_segmented_2nd=synth_powers_segmented_2nd,
                    )
                )

                res = minimize(fittingFunc, p0Firsts, bounds=boundsFirsts)
                res.fun
                fitsBoloFirsts.append(res.x[: self.numTorLocs])
                fitsBoloSeconds.append(res.x[self.numTorLocs :])
                preScales.append(preScaleFactor)

                degree_of_freedom = (
                    numChannels - (len(p0Firsts) + 1) - numIgnoredChannels
                )
                chisq = (res.fun) / degree_of_freedom
                chisqlist.append(chisq)
                pValList.append(RedChi2_To_Pvalue(chisq, degree_of_freedom))

            else:
                synth_powers_unsegmented = copy(radDist.boloCameras_powers)
                synth_powers_segmented = copy(radDist.bolo_powers_segmented)
                num_segments = len(synth_powers_segmented[0][0])

                # Creates an array of angles which correspond to the center of each segment.
                # first half are positive, second half are negative: goes from 0 to pi then
                # -pi to 0. To match how the powers array indexing works
                halfSegmentPhiWidth = np.pi / num_segments
                segmentCenterPhis = np.linspace(
                    halfSegmentPhiWidth,
                    (2.0 * np.pi) - halfSegmentPhiWidth,
                    num_segments,
                )
                for phiIndx in range(len(segmentCenterPhis)):
                    if segmentCenterPhis[phiIndx] > np.pi:
                        segmentCenterPhis[phiIndx] += -2 * np.pi

                # uniformly pre-scales synthetic powers to same order of magnitude as experimental
                # data, to put in range of fitting algorithm
                preScaleFactorNum = np.sum(exp_powers_unsegmented)
                preScaleFactorDenom = np.sum(synth_powers_unsegmented)
                preScaleFactor = preScaleFactorNum / preScaleFactorDenom
                # print("Pre Scale Factor = " + str(preScaleFactor))

                numIgnoredChannels = 0
                # this loop for counting the number of ignored channels for the purposes of the degree of freedom
                # and for uniformly rescaling the synthetic powers to start closer to the experimental values
                # currently, no channels are ignored
                for CameraIndx in range(len(exp_powers_unsegmented)):
                    for channelIndx in range(len(exp_powers_unsegmented[CameraIndx])):
                        channelExpPower = exp_powers_unsegmented[CameraIndx][
                            channelIndx
                        ]
                        if channelExpPower <= minPowerCutoff:
                            numIgnoredChannels += 1
                        for channelSegmentIndx in range(
                            len(synth_powers_segmented[CameraIndx][channelIndx])
                        ):
                            segmentSynthPower = synth_powers_segmented[CameraIndx][
                                channelIndx
                            ][channelSegmentIndx]
                            synth_powers_segmented[CameraIndx][channelIndx][
                                channelSegmentIndx
                            ] = (segmentSynthPower * preScaleFactor)

                if radDist.distType in ["ReflectedToroidal", "TiltedReflectedToroidal"]:
                    p0Firsts, p0Seconds, boundsFirsts, boundsSeconds, fittingFunc = (
                        self.fitting_func_setup_segmented_reflectedtor(
                            Bolo_exp_unsegmented=exp_powers_unsegmented,
                            Synth_powers_segmented=synth_powers_segmented,
                            PowerUpperBound=PowerUpperBound,
                            MinPowerCutoff=minPowerCutoff,
                            SegmentCenterPhis=segmentCenterPhis,
                            MaxExpPower=maxExpPower,
                        )
                    )
                elif radDist.distType in ["ElongatedRing"]:
                    p0Firsts, p0Seconds, boundsFirsts, boundsSeconds, fittingFunc = (
                        self.fitting_func_setup_segmented_ering(
                            Bolo_exp_unsegmented=exp_powers_unsegmented,
                            Synth_powers_segmented=synth_powers_segmented,
                            PowerUpperBound=PowerUpperBound,
                            MinPowerCutoff=minPowerCutoff,
                            SegmentCenterPhis=segmentCenterPhis,
                            MaxExpPower=maxExpPower,
                        )
                    )
                else:
                    p0Firsts, p0Seconds, boundsFirsts, boundsSeconds, fittingFunc = (
                        self.fitting_func_setup_segmented(
                            Bolo_exp_unsegmented=exp_powers_unsegmented,
                            Synth_powers_segmented=synth_powers_segmented,
                            PowerUpperBound=PowerUpperBound,
                            MinPowerCutoff=minPowerCutoff,
                            SegmentCenterPhis=segmentCenterPhis,
                            MaxExpPower=maxExpPower,
                        )
                    )

                res = minimize(
                    fittingFunc, p0Firsts, bounds=boundsFirsts
                )  # second punctures not yet involved
                res.fun
                fitsBoloFirsts.append(res.x[: self.numTorLocs])
                fitsBoloSeconds.append(res.x[self.numTorLocs :])
                preScales.append(preScaleFactor)

                degree_of_freedom = (
                    numChannels - (len(p0Firsts) + 1) - numIgnoredChannels
                )
                chisq = (res.fun) / degree_of_freedom
                chisqlist.append(chisq)
                pValList.append(RedChi2_To_Pvalue(chisq, degree_of_freedom))

        return (
            chisqlist,
            pValList,
            fitsBoloFirsts,
            fitsBoloSeconds,
            channel_errs,
            preScales,
        )

    def calc_fits(self, Etime, ErrorPool=False, PvalCutoff=None):
        # calculates reduced chi^2 values for radiation structure library for one timestep

        if len(self.allRadDistsVec) == 0:
            self.load_raddists(TokamakName=self.tokamakAMode.tokamakName)

        if self.comparingTo == "Experiment":
            boloExpData = self.load_bolo_exp_timestep(EvalTime=Etime)
        elif self.comparingTo == "Simulation":
            boloExpData = self.load_radDist_as_exp()[Etime]

        if self.torSegmented:
            (
                self.chisqVec,
                self.pvalVec,
                self.fitsBoloFirsts,
                self.fitsBoloSeconds,
                self.channel_errs,
                self.preScaleVec,
            ) = self.calc_pvals_segmented(self.allRadDistsVec, BoloExpData=boloExpData)
        else:
            (
                self.chisqVec,
                self.pvalVec,
                self.fitsBoloFirsts,
                self.fitsBoloSeconds,
                self.channel_errs,
                self.preScaleVec,
            ) = self.calc_pvals(self.allRadDistsVec, BoloExpData=boloExpData)

        # take minimum pval RadDist, which is the closest fit
        if (
            min(self.pvalVec) == 1.0
        ):  # covers case where Pvals bottom out (no pval better than 3). Not Ideal.
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
        if self.torSegmented:
            print(
                "best fit radDist at this timestep is toroidal z="
                + str(self.minRadDist.startZ)
                + ", r="
                + str(self.minRadDist.startR)
                + ", polsigma="
                + str(self.minRadDist.polSigma)
            )

        if ErrorPool:
            self.errorDists = []
            self.errorFitsFirsts = []
            self.errorFitsSeconds = []
            self.errorPrescales = []
            poolsize = 0
            for distNum in range(len(self.allRadDistsVec)):
                if self.pvalVec[distNum] <= PvalCutoff:
                    poolsize = poolsize + 1
                    self.errorDists.append(self.allRadDistsVec[distNum])
                    self.errorFitsFirsts.append(self.fitsBoloFirsts[distNum])
                    self.errorFitsSeconds.append(self.fitsBoloSeconds[distNum])
                    self.errorPrescales.append(self.preScaleVec[distNum])

            if (
                not self.errorDists
            ):  # covers case where even best fit radDist is not within Pval cutoff.
                # Adds just best fit to error pool
                self.errorDists.append(self.minRadDist)
                self.errorFitsFirsts.append(self.bestFitsBoloFirsts)
                self.errorFitsSeconds.append(self.bestFitsBoloSeconds)
                self.errorPrescales.append(self.minPreScale)
            print(" pool size is " + str(poolsize))

    def gaussian(self, Phi, Sigma, Mu, Amplitude, OffsetVert):
        coeff = 1.0 / (Sigma * math.sqrt(2.0 * math.pi))
        exponential = np.exp(-(((Phi - Mu) / Sigma) ** 2) / 2.0)
        return (Amplitude * coeff * exponential) + OffsetVert

    def gaussian_no_coeff(self, Phi, Sigma, Mu, Amplitude, OffsetVert):
        coeff = 1.0 / (Sigma * math.sqrt(2.0 * math.pi))
        return (
            self.gaussian(
                Phi=Phi, Sigma=Sigma, Mu=Mu, Amplitude=Amplitude, OffsetVert=OffsetVert
            )
            / coeff
        )

    def asymmetric_gaussian_arr(self, Phi, SigmaLeft, SigmaRight, Amplitude, Mu=None):
        # Returns an asymmetric gaussian. Implemented to handle arrays, since the curve_fit
        # function seems to need that

        if hasattr(Mu, "__len__"):
            mu = Mu
        elif Mu == None:
            mu = self.tokamakAMode.injectionPhiTor
        else:
            mu = Mu

        if hasattr(Phi, "__len__"):  # check if Phi is array-like
            yval = []
            # for i in range(len(Phi)):
            for phi in Phi:
                # phi = Phi[i]
                if phi < mu:
                    yval.append(
                        self.gaussian_no_coeff(
                            Phi=phi,
                            Sigma=SigmaLeft,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                    )
                else:
                    yval.append(
                        self.gaussian_no_coeff(
                            Phi=phi,
                            Sigma=SigmaRight,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                    )
        else:
            if Phi < mu:
                yval = self.gaussian_no_coeff(
                    Phi=Phi, Sigma=SigmaLeft, Mu=mu, Amplitude=Amplitude, OffsetVert=0.0
                )
            else:
                yval = self.gaussian_no_coeff(
                    Phi=Phi,
                    Sigma=SigmaRight,
                    Mu=mu,
                    Amplitude=Amplitude,
                    OffsetVert=0.0,
                )

        return yval

    def asymmetric_gaussian_extrapeaked_arr(
        self,
        Phi,
        SigmaLeft,
        SigmaRight,
        Amplitude,
        Mu=None,
        extrapeakMult=1,
        extrapeakWidth=(2.0 * math.pi / 8.0),
        HArrayPhi=(5.0 * math.pi / 4.0),
    ):
        # Returns an asymmetric gaussian. Implemented to handle arrays, since the curve_fit
        # function seems to need that
        # has extra peaking of extrapeakMult within extrapeakWidth of injector location

        injectorPhi = self.tokamakAMode.injectionPhiTor

        if hasattr(Mu, "__len__"):
            mu = Mu
        elif Mu == None:
            mu = self.tokamakAMode.injectionPhiTor
        else:
            mu = Mu

        if self.extrapeaked_variation == 1:
            if hasattr(Phi, "__len__"):  # check if Phi is array-like
                yval = []
                # for i in range(len(Phi)):
                for phi in Phi:
                    # phi = Phi[i]
                    if phi < mu:
                        newyval = self.gaussian_no_coeff(
                            Phi=phi,
                            Sigma=SigmaLeft,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                        if abs(phi - injectorPhi) <= extrapeakWidth:
                            newyval = newyval * extrapeakMult
                        yval.append(newyval)
                    else:
                        newyval = self.gaussian_no_coeff(
                            Phi=phi,
                            Sigma=SigmaRight,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                        if abs(phi - injectorPhi) <= extrapeakWidth:
                            newyval = newyval * extrapeakMult
                        yval.append(newyval)
            else:
                if Phi < mu:
                    yval = self.gaussian_no_coeff(
                        Phi=Phi,
                        Sigma=SigmaLeft,
                        Mu=mu,
                        Amplitude=Amplitude,
                        OffsetVert=0.0,
                    )
                else:
                    yval = self.gaussian_no_coeff(
                        Phi=Phi,
                        Sigma=SigmaRight,
                        Mu=mu,
                        Amplitude=Amplitude,
                        OffsetVert=0.0,
                    )
                if abs(Phi - injectorPhi) <= extrapeakWidth:
                    yval = yval * extrapeakMult
        elif self.extrapeaked_variation == 2:
            if hasattr(Phi, "__len__"):  # check if Phi is array-like
                yval = []
                # for i in range(len(Phi)):
                for phi in Phi:
                    if abs(phi - injectorPhi) <= extrapeakWidth:
                        hArrayVal1 = self.gaussian_no_coeff(
                            Phi=HArrayPhi,
                            Sigma=SigmaRight,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                        hArrayVal2 = self.gaussian_no_coeff(
                            Phi=(HArrayPhi - (2.0 * math.pi)),
                            Sigma=SigmaLeft,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                        newyval = (hArrayVal1 + hArrayVal2) * self.extrapeakedMult
                        yval.append(newyval)
                    else:
                        if phi < mu:
                            newyval = self.gaussian_no_coeff(
                                Phi=phi,
                                Sigma=SigmaLeft,
                                Mu=mu,
                                Amplitude=Amplitude,
                                OffsetVert=0.0,
                            )
                            yval.append(newyval)
                        else:
                            newyval = self.gaussian_no_coeff(
                                Phi=phi,
                                Sigma=SigmaRight,
                                Mu=mu,
                                Amplitude=Amplitude,
                                OffsetVert=0.0,
                            )
                            yval.append(newyval)
            else:
                if abs(Phi - injectorPhi) <= extrapeakWidth:
                    hArrayVal1 = self.gaussian_no_coeff(
                        Phi=HArrayPhi,
                        Sigma=SigmaRight,
                        Mu=mu,
                        Amplitude=Amplitude,
                        OffsetVert=0.0,
                    )
                    hArrayVal2 = self.gaussian_no_coeff(
                        Phi=(HArrayPhi - (2.0 * math.pi)),
                        Sigma=SigmaLeft,
                        Mu=mu,
                        Amplitude=Amplitude,
                        OffsetVert=0.0,
                    )
                    yval = (hArrayVal1 + hArrayVal2) * self.extrapeakedMult
                else:
                    if Phi < mu:
                        yval = self.gaussian_no_coeff(
                            Phi=Phi,
                            Sigma=SigmaLeft,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                    else:
                        yval = self.gaussian_no_coeff(
                            Phi=Phi,
                            Sigma=SigmaRight,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )

        return yval

    def cosine_tordist_1pi_arr(self, Phi, BaselineAmplitude, CosineAmpFrac, Mu=None):
        # Returns a cosine distribution from mu-pi to mu+pi. Implemented to handle arrays

        if hasattr(Mu, "__len__"):  # check if Mu is array-like
            mu = Mu
        elif Mu == None:
            mu = self.tokamakAMode.injectionPhiTor
        else:
            mu = Mu

        if hasattr(Phi, "__len__"):  # check if Phi is array-like
            yvals = []
            for phi in Phi:
                if abs(phi - mu) > math.pi:  # make sure cosine goes from mu-pi to mu+pi
                    yval = 0.0
            else:
                yval = BaselineAmplitude * (1 + (CosineAmpFrac * math.cos((phi - mu))))

            yvals.append(yval)
        else:
            phi = Phi
            if abs(phi - mu) > math.pi:  # make sure cosine goes from mu-pi to mu+pi
                yvals = 0.0
            else:
                yvals = BaselineAmplitude * (1 + (CosineAmpFrac * math.cos((phi - mu))))

        return yvals

    def cosine_tordist_2pi_arr(self, Phi, BaselineAmplitude, CosineAmpFrac, Mu=None):
        # Returns a cosine distribution from mu-pi to mu+pi. Implemented to handle arrays

        if hasattr(Mu, "__len__"):  # check if Mu is array-like
            mu = Mu
        elif Mu == None:
            mu = self.tokamakAMode.injectionPhiTor
        else:
            mu = Mu

        if hasattr(Phi, "__len__"):  # check if Phi is array-like
            yvals = []
            for phi in Phi:
                if abs(phi - mu) > math.pi:  # make sure cosine goes from mu-pi to mu+pi
                    if (phi - mu) > math.pi:
                        phi = phi - (2.0 * math.pi)
                    else:
                        phi = phi + (2.0 * math.pi)

            yval = BaselineAmplitude * (1 + (CosineAmpFrac * math.cos((phi - mu))))
            yvals.append(yval)
        else:
            phi = Phi
            if abs(phi - mu) > math.pi:  # make sure cosine goes from mu-pi to mu+pi
                if (phi - mu) > math.pi:
                    phi = phi - (2.0 * math.pi)
                else:
                    phi = phi + (2.0 * math.pi)

            yvals = BaselineAmplitude * (1 + (CosineAmpFrac * math.cos((phi - mu))))

        return yvals

    def triple_asymmetric_gaussian_tordist_arr(
        self, Phi, SigmaLeft, SigmaRight, Amplitude, Mu=None
    ):
        # Returns three asymmetric gaussians spaced by 120 degrees.
        # Used to model the "resonant" 6-injector mitigation case, in conjunction with
        # ReflectedToroidal radiation structures.
        # Implemented to handle arrays, since the curve_fit function seems to need that

        if hasattr(Mu, "__len__"):
            mu = Mu
        elif Mu == None:
            mu = self.tokamakAMode.injectionPhiTor
        else:
            mu = Mu

        if hasattr(Phi, "__len__"):  # check if Phi is array-like
            yval = []

            for phi in Phi:
                # shift phi to between mu - (math.pi/3.0) and mu + (math.pi/3.0)
                while phi > (mu + (math.pi / 3.0)):
                    phi = phi - (2.0 * math.pi / 3.0)

                while phi < (mu - (math.pi / 3.0)):
                    phi = phi + (2.0 * math.pi / 3.0)

                if phi < mu:
                    yval.append(
                        self.gaussian_no_coeff(
                            Phi=phi,
                            Sigma=SigmaLeft,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                    )
                else:
                    yval.append(
                        self.gaussian_no_coeff(
                            Phi=phi,
                            Sigma=SigmaRight,
                            Mu=mu,
                            Amplitude=Amplitude,
                            OffsetVert=0.0,
                        )
                    )
        else:
            phi = Phi

            # shift phi to between mu - (math.pi/3.0) and mu + (math.pi/3.0)
            while phi > (mu + (math.pi / 3.0)):
                phi = phi - (2.0 * math.pi / 3.0)

            while phi < (mu - (math.pi / 3.0)):
                phi = phi + (2.0 * math.pi / 3.0)

            if phi < mu:
                yval = self.gaussian_no_coeff(
                    Phi=phi, Sigma=SigmaLeft, Mu=mu, Amplitude=Amplitude, OffsetVert=0.0
                )
            else:
                yval = self.gaussian_no_coeff(
                    Phi=phi,
                    Sigma=SigmaRight,
                    Mu=mu,
                    Amplitude=Amplitude,
                    OffsetVert=0.0,
                )

        return yval

    def fit_asym_gaussian(
        self,
        PhiCoords=np.array([-2.0, -1.0, 1.0, 2.0]),
        FitYVals=np.array([3.0, 4.0, 3.0, 0.01]),
        ParametersGuess=np.array([1.0, 1.0, 2.2, np.nan]),
        MovePeak=False,
        PlotFit=False,
        MaxIters=800,
        JetPaperPlot=False,
    ):
        # for JetPaperPlot: PhiCoords = np.array([-3.0*np.pi/2.0, -3.0*np.pi/4.0, np.pi/2.0, 5.0*np.pi/4.0])
        # for JetPaperPlot: FitYVals = np.array([3.0, 4.0, 1.0, 0.01])

        if not MovePeak:
            parametersGuess = ParametersGuess[0:3]
            parameters = curve_fit(
                f=self.asymmetric_gaussian_arr,
                xdata=PhiCoords,
                ydata=FitYVals,
                p0=parametersGuess,
                bounds=[(0.0, 0.0, 0.0), (np.inf, np.inf, np.inf)],
                maxfev=MaxIters,
            )[0]

            sigmaLeft, sigmaRight, amplitude = (
                parameters[0],
                parameters[1],
                parameters[2],
            )
            mu = self.tokamakAMode.injectionPhiTor
        else:
            parametersGuess = ParametersGuess
            if np.isnan(ParametersGuess[3]):
                parametersGuess[3] = self.tokamakAMode.injectionPhiTor
            parameters = curve_fit(
                f=self.asymmetric_gaussian_arr,
                xdata=PhiCoords,
                ydata=FitYVals,
                p0=ParametersGuess,
                bounds=[(0.0, 0.0, 0.0, -np.pi), (np.inf, np.inf, np.inf, np.pi)],
                maxfev=MaxIters,
            )[0]

            sigmaLeft, sigmaRight, amplitude, mu = (
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
            )

        if PlotFit:
            plt.figure(figsize=(4.5, 3.0))
            xvalues = np.linspace(-2.0 * math.pi, 2.0 * math.pi, 100)
            yvalues = self.asymmetric_gaussian_arr(
                Phi=xvalues,
                SigmaLeft=sigmaLeft,
                SigmaRight=sigmaRight,
                Amplitude=amplitude,
                Mu=mu,
            )
            xaxisLabel = r"$\phi$ Toroidal [rad]"
            yaxisLabel = "Radiation Structure\nAmplitude"
            plt.xlabel(xaxisLabel)
            plt.ylabel(yaxisLabel)
            if JetPaperPlot:
                plt.scatter(
                    PhiCoords[2],
                    FitYVals[2],
                    color="dodgerblue",
                    label="Vert. Array Puncture 1",
                )
                plt.scatter(
                    PhiCoords[0],
                    FitYVals[0],
                    color="dodgerblue",
                    marker="^",
                    label="Vert. Array Puncture 2",
                )
                plt.plot(
                    xvalues, yvalues, color="orange", label="Asymmetric Gaussian Fit"
                )
                plt.scatter(
                    PhiCoords[1],
                    FitYVals[1],
                    color="limegreen",
                    label="Hor. Array Puncture 1",
                )
                plt.scatter(
                    PhiCoords[3],
                    FitYVals[3],
                    color="limegreen",
                    marker="^",
                    label="Hor. Array Puncture 2",
                )
                plt.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.42, -0.3))
                plt.tight_layout()
            else:
                plt.scatter(PhiCoords, FitYVals, label="Puncture Values")
                plt.plot(
                    xvalues, yvalues, color="orange", label="Asymmetric Gaussian Fit"
                )
                plt.legend()
            plt.show()

        return sigmaLeft, sigmaRight, amplitude, mu

    def fit_asym_gaussian_extrapeaked(
        self,
        PhiCoords=np.array([-2.0, -1.0, 1.0, 2.0]),
        FitYVals=np.array([3.0, 4.0, 3.0, 0.01]),
        ParametersGuess=np.array([1.0, 1.0, 2.2, np.nan]),
        MovePeak=False,
        PlotFit=False,
        MaxIters=800,
        JetPaperPlot=False,
    ):
        # for JetPaperPlot: PhiCoords = np.array([-3.0*np.pi/2.0, -3.0*np.pi/4.0, np.pi/2.0, 5.0*np.pi/4.0])
        # for JetPaperPlot: FitYVals = np.array([3.0, 4.0, 1.0, 0.01])

        if not MovePeak:
            parametersGuess = ParametersGuess[0:3]
            parameters = curve_fit(
                f=self.asymmetric_gaussian_extrapeaked_arr,
                xdata=PhiCoords,
                ydata=FitYVals,
                p0=parametersGuess,
                bounds=[(0.0, 0.0, 0.0), (np.inf, np.inf, np.inf)],
                maxfev=MaxIters,
            )[0]

            sigmaLeft, sigmaRight, amplitude = (
                parameters[0],
                parameters[1],
                parameters[2],
            )
            mu = self.tokamakAMode.injectionPhiTor
        else:
            parametersGuess = ParametersGuess
            if np.isnan(ParametersGuess[3]):
                parametersGuess[3] = self.tokamakAMode.injectionPhiTor
            parameters = curve_fit(
                f=self.asymmetric_gaussian_extrapeaked_arr,
                xdata=PhiCoords,
                ydata=FitYVals,
                p0=ParametersGuess,
                bounds=[(0.0, 0.0, 0.0, -np.pi), (np.inf, np.inf, np.inf, np.pi)],
                maxfev=MaxIters,
            )[0]

            sigmaLeft, sigmaRight, amplitude, mu = (
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
            )

        if PlotFit:

            xvalues = np.linspace(-2.0 * math.pi, 2.0 * math.pi, 100)
            yvalues = self.asymmetric_gaussian_extrapeaked_arr(
                Phi=xvalues,
                SigmaLeft=sigmaLeft,
                SigmaRight=sigmaRight,
                Amplitude=amplitude,
                Mu=mu,
            )
            xaxisLabel = r"$\phi$ Toroidal [rad]"
            yaxisLabel = "Radiation Structure\nAmplitude"
            if JetPaperPlot:
                plt.figure(figsize=(4.5, 3.0))
                plt.xlabel(xaxisLabel)
                plt.ylabel(yaxisLabel)
                plt.scatter(
                    PhiCoords[2],
                    FitYVals[2],
                    color="dodgerblue",
                    label="Vert. Array Puncture 1",
                )
                plt.scatter(
                    PhiCoords[0],
                    FitYVals[0],
                    color="dodgerblue",
                    marker="^",
                    label="Vert. Array Puncture 2",
                )
                plt.plot(
                    xvalues, yvalues, color="orange", label="Asymmetric Gaussian Fit"
                )
                plt.scatter(
                    PhiCoords[1],
                    FitYVals[1],
                    color="limegreen",
                    label="Hor. Array Puncture 1",
                )
                plt.scatter(
                    PhiCoords[3],
                    FitYVals[3],
                    color="limegreen",
                    marker="^",
                    label="Hor. Array Puncture 2",
                )
                plt.legend(ncol=2, loc="center right", bbox_to_anchor=(0.42, -0.3))
                plt.tight_layout()
            else:
                plt.figure(figsize=(4.5, 4.0))
                plt.xlabel(xaxisLabel)
                plt.ylabel(yaxisLabel)
                plt.scatter(PhiCoords, FitYVals, label="Puncture Values")
                plt.plot(xvalues, yvalues, color="orange", label="Toroidal Dist.")
                plt.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.0))
                plt.tight_layout()
            plt.show()
        return sigmaLeft, sigmaRight, amplitude, mu

    def calc_rad_error(self, PvalCutoff, MovePeak):
        # finds error bar ranges of rad power and tpf for a single timestep

        if not self.errorDists:
            raise Exception(
                "Must run calc_fits with ErrorPool=True before calc_rad_error"
            )

        errorRadPowers = []
        errorTpfs = []
        for distNum in range(len(self.errorDists)):
            try:
                errorRadPower, errorExtrasPowers, errorTpf, errorToroidalFitCoeffs = (
                    self.calc_rad_power(
                        RadDist=self.errorDists[distNum],
                        PreScale=self.errorPrescales[distNum],
                        FitsFirsts=self.errorFitsFirsts[distNum],
                        FitsSeconds=self.errorFitsSeconds[distNum],
                        MovePeak=MovePeak,
                    )
                )
                errorRadPowers.append(errorRadPower)
                errorTpfs.append(errorTpf)
            except:
                print(
                    "The toroidal distribution fitting of \
                      a helical distribution in pool of reasonable fits \
                      was unable to converge at this time"
                )

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

    def calc_tot_rad(
        self,
        TimePerStep,
        NumTimes=11,
        ErrorPool=False,
        PvalCutoff=0.9545,
        MovePeak=False,
    ):
        # finds total radiated energy for entire shot. Stores timestep-by-timestep best fit
        # information in the process, and error bound information if ErrorPool=True

        self.radPowers = []
        self.extrasPowers = []
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

            self.calc_fits(
                Etime=self.radPowerTimes[timeIndex],
                ErrorPool=ErrorPool,
                PvalCutoff=PvalCutoff,
            )

            print("Best Fit RadDist Type = " + self.minDistType)
            self.minRadDistList.append(self.minRadDist)
            self.minPreScaleList.append(self.minPreScale)
            self.minFitsFirsts.append(self.bestFitsBoloFirsts)
            self.minFitsSeconds.append(self.bestFitsBoloSeconds)
            self.minchisqList.append(self.minchisq)
            self.minpvalList.append(self.minpval)

            radPower, extrasPowers, tpf, torFitCoeffs = self.calc_rad_power(
                RadDist=self.minRadDist,
                PreScale=self.minPreScale,
                FitsFirsts=self.bestFitsBoloFirsts,
                FitsSeconds=self.bestFitsBoloSeconds,
                MovePeak=MovePeak,
            )

            self.radPowers.append(radPower)
            self.extrasPowers.append(extrasPowers)
            self.tpfs.append(tpf)
            self.minTorFitCoeffs.append(torFitCoeffs)

            print("Rad Power in GW is " + str(radPower * 1e-9))
            print("TPF is " + str(tpf))

            if ErrorPool:
                self.calc_rad_error(PvalCutoff=PvalCutoff, MovePeak=MovePeak)

            wrad = wrad + (radPower * TimePerStep)
            upperwrad = upperwrad + (self.upperBoundPower * TimePerStep)
            lowerwrad = lowerwrad + (self.lowerBoundPower * TimePerStep)
            self.accumWrads.append(wrad)
            self.upperAccumWrads.append(upperwrad)
            self.lowerAccumWrads.append(lowerwrad)

        self.totWrad = self.accumWrads[-1]
        self.upperTotWrad = self.upperAccumWrads[-1]
        self.lowerTotWrad = self.lowerAccumWrads[-1]

    def output_to_excel(self):
        # Output data from Emis3D as an excel file

        import pandas as pd

        # choose which variables to output to file, and what names they get in excel file
        vars_to_print = [
            "radPowerTimes",
            "minchisqList",
            "lowerBoundRadPowers",
            "radPowers",
            "upperBoundRadPowers",
            "lowerBoundTpfs",
            "tpfs",
            "upperBoundTpfs",
            "minRadDistList",
            "minRadDistList",
            "minRadDistList",
            "minRadDistList",
            "minRadDistList",
        ]
        excel_names = [
            "Time (s)",
            "Best Chisq",
            "Prad Lower (GW)",
            "Prad Best (GW)",
            "Prad Upper (GW)",
            "TPF Lower",
            "TPF Best",
            "TPF Upper",
            "Best Fit Type",
            "polSigma",
            "startR",
            "startZ",
            "elongation",
        ]

        # copy all attributes of Emis3D instance as dictionary
        varsDict = copy(vars(self))

        # make new dictionary with only desired variables, renamed and ordered as excel_names
        outputDict = {}
        for name in excel_names:
            values = varsDict[vars_to_print[excel_names.index(name)]]
            # also convert radiated powers from W to GW
            if name in ["Prad Lower (GW)", "Prad Best (GW)", "Prad Upper (GW)"]:
                values = [x * 1e-9 for x in values]
            elif name in ["Best Fit Type"]:
                values = [x.distType for x in values]
            elif name in ["polSigma", "startR", "startZ", "elongation"]:
                newvalues = []
                for x in values:
                    if hasattr(x, name):
                        newvalues.append(getattr(x, name))
                    else:
                        newvalues.append(None)
                values = newvalues
            outputDict[name] = values

        # convert to pandas dataframe
        outputDf = pd.DataFrame.from_dict(outputDict)

        # output as excel
        writer = pd.ExcelWriter(
            join(self.data_output_directory, str(self.shotnumber) + ".xlsx")
        )
        outputDf.to_excel(writer, sheet_name=str(self.shotnumber), startcol=0)

        # add any additional tokamak-specific variables to dictionary
        try:
            outputExtrasDict = self.tok_specific_output_data()
            outputExtrasDf = pd.DataFrame.from_dict(outputExtrasDict)
            outputExtrasDf.to_excel(
                writer, sheet_name=str(self.shotnumber), startcol=len(excel_names) + 2
            )
        except:
            pass

        # close excel writer
        writer.close()

    def plot_radDist_array(
        self, PlotData, Levels, Lengthx, Lengthy, Etime, ColorBarLabel, PlotDistType
    ):

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
        ax.set_aspect("equal")
        ax.set_xlabel("$R$ (m)")
        ax.set_ylabel("$Z$ (m)")

        # Create color contour of fits
        tcf = ax.tricontourf(
            plotRVecs, plotZVecs, PlotData, levels=Levels, cmap="gist_stern"
        )
        ax.plot(plotRVecs, plotZVecs, "xk")

        # Add white x and circle around best fit
        if self.minDistType == PlotDistType:
            bestRadDist = self.allRadDistsVec[self.minInd]
            r = bestRadDist.startR
            z = bestRadDist.startZ
            ax.plot(
                r,
                z,
                "o",
                markeredgecolor="white",
                fillstyle="none",
                markeredgewidth=3,
                markersize=40,
            )
            ax.plot(r, z, "x", markeredgecolor="white")

        # Set plot dimensions
        ax.set_xlim(
            [
                self.tokamakAMode.majorRadius - (1.1 * self.tokamakAMode.minorRadius),
                self.tokamakAMode.majorRadius + (1.1 * self.tokamakAMode.minorRadius),
            ]
        )
        ax.set_ylim(
            [
                (-2.4 * self.tokamakAMode.minorRadius),
                (2.4 * self.tokamakAMode.minorRadius),
            ]
        )

        # Set plot title
        ax.set_title(str(PlotDistType) + "s")

        # Set colorbar
        fig.colorbar(tcf, ax=ax, label=ColorBarLabel, shrink=0.5, format="%d")

        # Plot first wall curve
        r = self.tokamakAMode.wallcurve.vertices[:, 0]
        z = self.tokamakAMode.wallcurve.vertices[:, 1]
        ax.plot(r, z, "orange")

        # Plot Q=1.1,2, and 3 surfaces
        # r, z = self.tokamakAMode.get_qsurface_contour(
        #    Shotnumber=self.shotnumber, EvalTime=Etime - 5e-4, Qval=1.1
        # )
        ax.plot(r, z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(
            Shotnumber=self.shotnumber, EvalTime=Etime - 5e-4, Qval=2.0
        )
        ax.plot(r, z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(
            Shotnumber=self.shotnumber, EvalTime=Etime - 5e-4, Qval=3.0
        )
        ax.plot(r, z, "cyan")
        r, z = self.tokamakAMode.get_qsurface_contour(
            Shotnumber=self.shotnumber, EvalTime=Etime - 5e-4, Qval="Separatrix"
        )
        ax.plot(r, z, "cyan")

        return fig

    def plot_fits_array(
        self, Lengthx=3, Lengthy=4, Etime=50.89, PlotDistType="Helical"
    ):

        if not self.pvalVec:
            raise Exception("Must calculate p values before plotting fits")

        # Pull only RadDists of given type to be plotted
        plotChisq = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                plotChisq.append(self.chisqVec[distNum])

        levels = np.linspace(0, 40.0, num=20)

        fig = self.plot_radDist_array(
            PlotData=plotChisq,
            Levels=levels,
            Lengthx=Lengthx,
            Lengthy=Lengthy,
            Etime=Etime,
            ColorBarLabel="$\chi^2_r$",
            PlotDistType=PlotDistType,
        )

        return fig

    def plot_powers_array(
        self, Lengthx=3, Lengthy=4, Etime=50.89, PlotDistType="Helical", MovePeak=False
    ):

        if not self.pvalVec:
            raise Exception("Must calculate p values before plotting powers")

        # Pull only RadDists of given type to be plotted
        plotDistsRadPowers = []
        for distNum in range(len(self.allRadDistsVec)):
            if self.allRadDistsVec[distNum].distType == PlotDistType:
                thisRadDist = self.allRadDistsVec[distNum]

                radPower = self.calc_rad_power(
                    RadDist=thisRadDist,
                    PreScale=self.preScaleVec[distNum],
                    FitsFirsts=self.fitsBoloFirsts[distNum],
                    FitsSeconds=self.fitsBoloSeconds[distNum],
                    MovePeak=MovePeak,
                )[0]

                plotDistsRadPowers.append(radPower)

        bestFitRadDist = self.allRadDistsVec[self.minInd]
        bestFitRadPower = self.calc_rad_power(
            RadDist=bestFitRadDist,
            PreScale=self.preScaleVec[self.minInd],
            FitsFirsts=self.fitsBoloFirsts[self.minInd],
            FitsSeconds=self.fitsBoloSeconds[self.minInd],
            MovePeak=MovePeak,
        )[0]
        print("Best Fit Rad Power is " + str(bestFitRadPower / 1e9) + "GW")

        # This step removes very high outliers that screw up the plot
        # medPower = np.median(plotDistsRadPowers)

        for i in range(len(plotDistsRadPowers)):
            if plotDistsRadPowers[i] > (10.0 * bestFitRadPower):
                plotDistsRadPowers[i] = 0.0
            elif np.isnan(plotDistsRadPowers[i]):
                plotDistsRadPowers[i] = 0.0

        levels = np.linspace(0, max(plotDistsRadPowers) / 1e9, num=20)

        fig = self.plot_radDist_array(
            PlotData=np.array(plotDistsRadPowers) / 1e9,
            Levels=levels,
            Lengthx=Lengthx,
            Lengthy=Lengthy,
            Etime=Etime,
            ColorBarLabel="$P_{rad}$ (GW)",
            PlotDistType=PlotDistType,
        )

        return fig

    def plot_fits_channels(self, Etime=50.89, AsBrightness=True):

        bolo_exp, synthArrayFirst, synthArraySecond = self.plot_fits_data_organization(
            Etime=Etime
        )

        fig, axs = plt.subplots(self.numTorLocs, 1, figsize=(3, 2 * self.numTorLocs))

        channel_errs = self.channel_errs

        for numTor in range(self.numTorLocs):
            ax = axs[numTor]

            # Remove error bars that are larger than measurement itself, just for plotting
            for errIndex in range(len(channel_errs[numTor])):
                try:
                    if abs(channel_errs[numTor][errIndex]) > abs(
                        bolo_exp[numTor][errIndex]
                    ):
                        channel_errs[numTor][errIndex] = 0.0
                except:
                    pass

            # convert to power per m^2
            if AsBrightness:
                for i in range(len(bolo_exp[numTor])):
                    bolo_exp[numTor][i] = (
                        bolo_exp[numTor][i]
                        * 4.0
                        * math.pi
                        / self.bolo_etendues[numTor][i]
                    )
                    synthArrayFirst[numTor][i] = (
                        synthArrayFirst[numTor][i]
                        * 4.0
                        * math.pi
                        / self.bolo_etendues[numTor][i]
                    )
                    # only works if there is a second puncture, otherwise length of list is wrong
                    try:
                        synthArraySecond[numTor][i] = (
                            synthArraySecond[numTor][i]
                            * 4.0
                            * math.pi
                            / self.bolo_etendues[numTor][i]
                        )
                    except:
                        pass

                # for MW / m^2
                ax.set_ylabel(r"Power Location " + str(numTor) + " $(MW/m^2)$")
                bolo_exp[numTor] = [x * 1e-6 for x in bolo_exp[numTor]]
                channel_errs[numTor] = [x * 1e-6 for x in channel_errs[numTor]]
                synthArrayFirst[numTor] = [x * 1e-6 for x in synthArrayFirst[numTor]]
                try:
                    synthArraySecond[numTor] = [
                        x * 1e-6 for x in synthArraySecond[numTor]
                    ]
                except:
                    pass

            else:
                # for watts
                multiplier = 1.0
                ax.set_ylabel(r"Power Location " + str(numTor) + " $(W)$")
                # for milliwatts
                # multiplier =1e3
                # ax.set_ylabel(r"Power Location " + str(numTor) + " $(mW)$")

                bolo_exp[numTor] = [x * multiplier for x in bolo_exp[numTor]]
                channel_errs[numTor] = [x * multiplier for x in channel_errs[numTor]]
                synthArrayFirst[numTor] = [
                    x * multiplier for x in synthArrayFirst[numTor]
                ]
                synthArraySecond[numTor] = [
                    x * multiplier for x in synthArraySecond[numTor]
                ]

            channelNum = range(len(bolo_exp[numTor]))
            try:
                ax.errorbar(
                    channelNum,
                    bolo_exp[numTor],
                    yerr=channel_errs[numTor],
                    label="Experiment",
                    color="blue",
                )
            except:
                ax.plot(channelNum, bolo_exp[numTor], label="Experiment", color="blue")
            ax.plot(
                channelNum,
                synthArrayFirst[numTor],
                "-s",
                label="1st Puncture",
                color="gray",
            )
            try:
                ax.plot(
                    channelNum,
                    synthArraySecond[numTor],
                    "-x",
                    label="2nd Puncture",
                    color="gray",
                )
                ax.plot(
                    channelNum,
                    (
                        [
                            synthArrayFirst[numTor][i] + synthArraySecond[numTor][i]
                            for i in range(len(synthArrayFirst[numTor]))
                        ]
                    ),
                    "-o",
                    label="Full Synthetic",
                    color="green",
                )
            except:
                pass
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        plt.tight_layout()

        return fig

    def save_bolos_contour_plot(
        self,
        Times,
        Bolo_vals,
        Title,
        SaveName,
        SaveFolder,
        LowerBound=4,
        UpperBound=8.2,
        NumTicks=22,
    ):

        channel_list = np.array(range(len(Bolo_vals)))
        x, y = np.meshgrid(Times, channel_list)

        plot_vals = copy(Bolo_vals)
        for row in range(len(plot_vals)):
            for indx in range(len(plot_vals[row])):
                if plot_vals[row][indx] > 0:
                    plot_vals[row][indx] = np.log10(plot_vals[row][indx])
                else:
                    plot_vals[row][indx] = np.nan

        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(
            x,
            y,
            plot_vals,
            levels=np.linspace(LowerBound, UpperBound, NumTicks),
            cmap="jet",
        )
        colorbar = fig.colorbar(cp)
        colorbar.ax.set_ylabel("Log(Brightness [$W/m^2$])")
        ax.set_title(Title)
        ax.set_ylabel("Channel Number")
        ax.set_xlabel("Time [s]")
        savefile = join(SaveFolder, SaveName) + ".png"
        plt.savefig(savefile, format="png")

        plt.close(fig)

    def save_synth_contour_plot(
        self,
        ArrayNum,
        SaveName,
        SaveFolder,
        PreviousArrayNum=None,
        NumChannels=None,
        LowerBound=1e4,
    ):

        numTimes = len(self.radPowerTimes)
        if NumChannels == None:
            numChannels = 0
            bolo_powers_1st = self.rearrange_powers_array(
                self.minRadDistList[0].boloCameras_powers
            )
            if hasattr(bolo_powers_1st[ArrayNum][0], "len"):
                for subCameraNum in range(len(bolo_powers_1st[ArrayNum])):
                    numChannels += len(bolo_powers_1st[ArrayNum][subCameraNum])
            else:
                numChannels += len(bolo_powers_1st[ArrayNum])
        else:
            numChannels = NumChannels

        etendues = self.bolo_etendues[ArrayNum]
        bolos = np.zeros([numTimes, numChannels])
        for timeIndx in range(numTimes):
            plotFitsFirsts = self.minFitsFirsts[timeIndx]
            plotFitsSeconds = self.minFitsSeconds[timeIndx]
            preScale = self.minPreScaleList[timeIndx]
            minRadDist = copy(self.minRadDistList[timeIndx])
            for channelIndx in range(numChannels):
                boloValue = 0.0
                bolo_powers = self.rearrange_powers_array(
                    self.minRadDistList[timeIndx].boloCameras_powers
                )
                bolo_powers_2nd = self.rearrange_powers_array(
                    self.minRadDistList[timeIndx].boloCameras_powers_2nd
                )
                if hasattr(bolo_powers[ArrayNum][0], "len"):
                    for subCameraNum in range(len(bolo_powers[ArrayNum])):
                        boloValue = (
                            bolo_powers[ArrayNum][subCameraNum][channelIndx]
                            * preScale
                            * plotFitsFirsts[ArrayNum]
                        )
                        if (
                            hasattr(bolo_powers_2nd[ArrayNum][subCameraNum], "len")
                            and PreviousArrayNum != None
                        ):
                            boloValue += (
                                bolo_powers_2nd[ArrayNum][subCameraNum][channelIndx]
                                * preScale
                                * plotFitsFirsts[PreviousArrayNum]
                                * plotFitsSeconds[ArrayNum]
                            )
                else:
                    boloValue = (
                        bolo_powers[ArrayNum][channelIndx]
                        * preScale
                        * plotFitsFirsts[ArrayNum]
                    )
                    if (
                        hasattr(minRadDist.boloCameras_powers_2nd[ArrayNum], "len")
                        and PreviousArrayNum != None
                    ):
                        boloValue += (
                            bolo_powers_2nd[ArrayNum][channelIndx]
                            * preScale
                            * plotFitsFirsts[PreviousArrayNum]
                            * plotFitsSeconds[ArrayNum]
                        )
                boloValue = boloValue * 4.0 * math.pi / etendues[channelIndx]
                bolos[timeIndx][channelIndx] = boloValue

        # sets lower bound to plot values
        for i in range(numTimes):
            for j in range(numChannels):
                if bolos[i][j] < LowerBound:
                    bolos[i][j] = LowerBound

        bolos = bolos.T

        self.save_bolos_contour_plot(
            Times=self.radPowerTimes,
            Bolo_vals=bolos,
            Title="Array "
            + str(ArrayNum)
            + " Best Fit Brightnesses\nDischarge "
            + str(self.shotnumber),
            SaveName=SaveName,
            SaveFolder=SaveFolder,
        )

    def save_exp_contour_plot(
        self,
        ArrayNum,
        Title,
        SaveName,
        SaveFolder,
        StartTime=None,
        EndTime=None,
        EndChannel=None,
        DeleteChannels=None,
    ):

        if StartTime == None and EndTime == None:
            startTime = self.radPowerTimes[0]
            endTime = self.radPowerTimes[-1]
        else:
            startTime = StartTime
            endTime = EndTime

        if EndChannel != None:
            expData = self.load_bolo_exp_timerange(
                StartTime=startTime, EndTime=endTime, AsBrightnesses=True
            )[ArrayNum][:EndChannel]
        else:
            expData = self.load_bolo_exp_timerange(
                StartTime=startTime, EndTime=endTime, AsBrightnesses=True
            )[ArrayNum]

        expTimebase = self.load_bolo_timebase_range(
            StartTime=startTime, EndTime=endTime
        )[ArrayNum]

        if DeleteChannels != None:
            for indx in range(len(DeleteChannels)):
                channel = DeleteChannels[indx]
                expData[channel] = np.array([np.nan] * len(expData[channel]))

        self.save_bolos_contour_plot(
            Times=expTimebase,
            Bolo_vals=expData,
            Title=Title,
            SaveName=SaveName,
            SaveFolder=SaveFolder,
        )

    def save_sim_contour_plot(
        self,
        ArrayNum,
        Title,
        SaveName,
        SaveFolder,
        EndChannel=None,
        DeleteChannels=None,
    ):

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
                boloValue = (
                    boloValue
                    * 4.0
                    * math.pi
                    / self.bolo_etendues[ArrayNum][channelIndx]
                )
                simData[timeIndx][channelIndx] = boloValue

        simData = np.array(simData).T

        simTimebase = self.radPowerTimes

        if DeleteChannels != None:
            for indx in range(len(DeleteChannels)):
                channel = DeleteChannels[indx]
                simData[channel] = np.array([np.nan] * len(simData[channel]))

        self.save_bolos_contour_plot(
            Times=simTimebase,
            Bolo_vals=simData,
            Title=Title,
            SaveName=SaveName,
            SaveFolder=SaveFolder,
        )

    def make_crossSec_movie(self, Phi=None, MovePeak=False):

        import cv2

        if Phi == None:
            Phi = self.tokamakAMode.injectionPhiTor

        self.load_tokamak(Mode="Build")
        numFillZeros = len(str(len(self.minRadDistList)))
        for radDistNum in range(len(self.minRadDistList)):
            savefile = (
                join(self.videos_output_directory, "crossSecImg")
                + str(radDistNum).zfill(numFillZeros)
                + ".png"
            )
            radDist = copy(self.minRadDistList[radDistNum])
            radDist.set_tokamak(self.tokamakBMode)
            radDist.make_build_mode()

            if radDist.distType == "Helical":

                sigmaLeft, sigmaRight, amplitude, mu = (
                    self.find_asym_gaussian_parameters(
                        FitsFirsts=self.minFitsFirsts[radDistNum],
                        FitsSeconds=self.minFitsSeconds[radDistNum],
                        MovePeak=MovePeak,
                    )
                )

                def torDistFunc(Phi0):
                    val = self.asymmetric_gaussian_arr(
                        Phi=Phi0,
                        SigmaLeft=sigmaLeft,
                        SigmaRight=sigmaRight,
                        Amplitude=amplitude,
                        Mu=mu,
                    )
                    return val

                crossSecPlot = radDist.plot_crossSec(Phi=Phi, TorDistFunc=torDistFunc)
            else:
                crossSecPlot = radDist.plot_crossSec(Phi=Phi)

            crossSecPlot.savefig(savefile, format="png")
            plt.close(crossSecPlot)

        video_name = join(self.videos_output_directory, "crossSecs.avi")

        images = [
            img
            for img in listdir(self.videos_output_directory)
            if img.endswith(".png") and img.startswith("crossSecImg")
        ]
        images.sort()
        frame = cv2.imread(join(self.videos_output_directory, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(join(self.videos_output_directory, image)))

        for radDistNum in range(len(self.minRadDistList)):
            savefile = (
                join(self.videos_output_directory, "crossSecImg")
                + str(radDistNum).zfill(numFillZeros)
                + ".png"
            )
            remove(savefile)

        video.release()

    def make_unwrapped_movie(
        self, Resolution=30, Alpha=0.005, SpotSize=20, MovePeak=False
    ):
        import cv2

        self.load_tokamak(Mode="Build")
        numFillZeros = len(str(len(self.minRadDistList)))

        for radDistNum in range(len(self.minRadDistList)):
            savefile = (
                join(self.videos_output_directory, "radDistImg")
                + str(radDistNum).zfill(numFillZeros)
                + ".png"
            )
            radDist = copy(self.minRadDistList[radDistNum])
            radDist.set_tokamak(self.tokamakBMode)
            radDist.make_build_mode()

            if radDist.distType == "Helical":

                sigmaLeft, sigmaRight, amplitude, mu = (
                    self.find_asym_gaussian_parameters(
                        FitsFirsts=self.minFitsFirsts[radDistNum],
                        FitsSeconds=self.minFitsSeconds[radDistNum],
                        MovePeak=MovePeak,
                    )
                )

                def torDistFunc(Phi0):
                    val = self.asymmetric_gaussian_arr(
                        Phi=Phi0,
                        SigmaLeft=sigmaLeft,
                        SigmaRight=sigmaRight,
                        Amplitude=amplitude,
                        Mu=mu,
                    )
                    return val

                radDistPlot = radDist.plot_unwrapped(
                    TorDistFunc=torDistFunc,
                    SpotSize=SpotSize,
                    Resolution=Resolution,
                    Alpha=Alpha,
                )
            else:
                radDistPlot = radDist.plot_unwrapped(
                    SpotSize=SpotSize, Resolution=Resolution, Alpha=Alpha
                )

            radDistPlot.savefig(savefile, format="png")
            plt.close(radDistPlot)

        video_name = join(self.videos_output_directory, "unwrapped.avi")

        images = [
            img
            for img in listdir(self.videos_output_directory)
            if img.endswith(".png") and img.startswith("radDistImg")
        ]
        images.sort()
        frame = cv2.imread(join(self.videos_output_directory, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(join(self.videos_output_directory, image)))

        for radDistNum in range(len(self.minRadDistList)):
            savefile = (
                join(self.videos_output_directory, "radDistImg")
                + str(radDistNum).zfill(numFillZeros)
                + ".png"
            )
            remove(savefile)

        video.release()
