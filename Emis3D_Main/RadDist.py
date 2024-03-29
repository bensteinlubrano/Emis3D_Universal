# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:12:06 2021

@author: bemst
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import sys
from os.path import join
import json
from Util import XY_To_RPhi, RPhi_To_XY
from Diagnostic import Synth_Brightness_Observer

# raysect dependencies
from raysect.core.math import translate
from raysect.core.math.random import seed as raysectseed
from raysect.optical.material import VolumeTransform
from raysect.primitive import Cylinder

from cherab.tools.emitters import RadiationFunction

from copy import deepcopy

class RadDist(object):
    
    def __init__(self, NumBins = 18, NumPuncs = 2, Tokamak = None,\
                Mode = "Analysis", LoadFileName = None, SaveFileFolder=None,\
                TorSegmented=False):
        #[creates radiation distribution from a given evaluate function]
        
        self.tokamak = Tokamak
        self.saveFileFolder=SaveFileFolder
        self.torSegmented=TorSegmented
        self.randSeed = 10 # arbitrary. Just always using the same one so results don't vary

        if LoadFileName == None:
            self.numBins = NumBins
            self.numPuncs = NumPuncs
            self.powerPerBin = []
            self.boloCameras_powers = []
            self.boloCameras_powers_2nd = []

            # extras bolometers are bolos that are not used in the Emis3D fitting process, but their
            # values at each timestep according to the best fit/close fit pool radiation structures
            # are still recorded. E.g. the KB1s on JET
            self.boloCameras_powers_extras = []
            self.boloCameras_powers_extras_2nd = []

            if self.torSegmented:
                # structure of segmented bolo powers lists is:
                # cameras -> channels in each camera -> toroidal segments for each channel
                self.bolo_powers_segmented = []
                self.bolo_powers_segmented_2nd = []
                self.bolo_powers_extras_segmented = []
                self.bolo_powers_extras_segmented_2nd = []

                # this is a bin index used in iterating over the bins to do a segmented
                # observation. Necessary because the observe function takes functions of
                # strictly three parameters, so this can't be passed in-line
                self.tempObserverBin = None
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.numBins = properties["numBins"]
            self.numPuncs = properties["numPuncs"]
            self.boloCameras_powers = properties["boloCameras_powers"]
            self.boloCameras_powers_2nd = properties["boloCameras_powers_2nd"]
            if "boloCameras_powers_extras" in properties:
                self.boloCameras_powers_extras = properties["boloCameras_powers_extras"]
                self.boloCameras_powers_extras_2nd = properties["boloCameras_powers_extras_2nd"]
            self.powerPerBin = properties["powerPerBin"]
            if self.torSegmented:
                try:
                    self.bolo_powers_segmented = properties["bolo_powers_segmented"]
                    self.bolo_powers_segmented_2nd = properties["bolo_powers_segmented_2nd"]
                    if "bolo_powers_extras_segmented" in properties:
                        self.bolo_powers_extras_segmented = properties["bolo_powers_extras_segmented"]
                        self.bolo_powers_extras_segmented_2nd = properties["bolo_powers_extras_segmented_2nd"]

                    self.tempObserverBin = None
                except:
                    errStatement = """ Attempted to load a radiation structure with unsegmented
                        bolo powers as a segmented radiation structure"""
                    print(errStatement)
                    sys.exit(1)

    def set_tokamak(self, Tokamak):
        self.tokamak = Tokamak
    
    def prepare_for_JSON(self, properties):
        
        # JSON doesn't know how to deal with objects; just save names to
        # rebuild if necessary
        if self.torSegmented:
            properties.pop("tempObserverBin")
        
        properties["tokamakName"] = self.tokamak.tokamakName
        properties.pop("tokamak")
        properties.pop("torSegmented")
        
        return properties
    
    def build(self):
        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass Tokamak = None and Mode = 'Build'\
                  to this RadDist")
            sys.exit(1)
            
        self.power_per_bin_calc()
        self.bolos_observe()
        self.save_RadDist()
        try:
            self.save_RadDist_Complete()
        except:
            pass
    
    def evaluate_first_punc(self, X,Y,Z):
        # is the 0 in phi<0 here hardcoded to SPARC injector location? Probably.
        # if we ever go to multiple injector Emis3D fitting, will need to adjust
        if self.torSegmented:
            r,phi = XY_To_RPhi(X,Y)
            if(phi < 0):
               phi =  phi + 2*np.pi
            t = self.tempObserverBin
            inc = np.pi/((self.numBins/2))
            binPhi = inc * t
            if(phi >= binPhi and phi < (binPhi + inc)):
                return self.evaluate(X,Y,Z, EvalFirstPunc=1, EvalSecondPunc=0)
            else:
                return 0    
        else:
            return self.evaluate(X,Y,Z, EvalFirstPunc=1, EvalSecondPunc=0)
    
    def evaluate_second_punc(self, X,Y,Z):
        if self.torSegmented:
            r,phi = XY_To_RPhi(X,Y)
            if(phi < 0):
              phi =  phi + 2*np.pi
            t = self.tempObserverBin
            inc = np.pi/((self.numBins/2))
            binPhi = inc * t
            if(phi >= binPhi and phi < (binPhi + inc)):
                return self.evaluate(X,Y,Z, EvalFirstPunc=0, EvalSecondPunc=1)
            else:
                return 0
        else:    
            return self.evaluate(X,Y,Z, EvalFirstPunc=0, EvalSecondPunc=1)
        
    def evaluate_both_punc(self, X,Y,Z):
        return self.evaluate_first_punc(X,Y,Z) + self.evaluate_second_punc(X,Y,Z)
        
    def evaluate_first_punc_cherab(self, X,Y,Z):
        # Various tokamaks use different toroidal angle conventions than Cherab. Emis3D uses the Cherab
        # Angle convention. self.tokamak.torConventionPhi accounts for this difference

        r, phi = XY_To_RPhi(X,Y)
        x,y = RPhi_To_XY(r, phi - self.tokamak.torConventionPhi)
                
        return self.evaluate_first_punc(x,y,Z)
    
    def evaluate_second_punc_cherab(self, X,Y,Z):
        
        r, phi = XY_To_RPhi(X,Y)
        x,y = RPhi_To_XY(r, phi - self.tokamak.torConventionPhi)
        
        return self.evaluate_second_punc(x,y,Z)
    
    def evaluate_both_punc_cherab(self, X,Y,Z):
        r, phi = XY_To_RPhi(X,Y)
        x,y = RPhi_To_XY(r, phi - self.tokamak.torConventionPhi)
        evalBoth = self.evaluate_first_punc(x,y,Z) + self.evaluate_second_punc(x,y,Z)
        return evalBoth
    
    def power_per_bin_calc(self, Errfrac = 0.01, Pointsupdate = int(1e5)):
        # Calculates total radiated power per solid angle emitted inside
        # a toroidal region with emissivity function from evaluate function

        # set random seed to always be the same so that the result is not different every run
        random.seed(self.randSeed)

        # Various possible errors
        if (self.numPuncs != 1) and (self.numPuncs != 2):
            print("Radiation integration code is only designed for one or two\
                  \npunctures (toroidal cycles). For values between one and two,\
                      \nuse two.")
            sys.exit(1)
            
        if not isinstance(self.numBins, int) or self.numBins < 1 or self.numBins > 360:
            print("Number of toroidal bins must be an integer between 1 and 360.")
            sys.exit(1)
            
        if Errfrac <= 0:
            print("Desired error must be a positive number.")
            sys.exit(1)
            
        if self.tokamak.volume <= 0:
            print("Volume of first wall region must be a positive number (in meters^3).")
            sys.exit(1)
            
        if not isinstance(Pointsupdate, int):
            print("Update frequency ('Pointsupdate') must be an integer greater than 0.")
            sys.exit(1)
            
        if Pointsupdate < 1e4 or Pointsupdate > 1e7:
            print("Warning: Update frequency ('Pointsupdate') is in an unusual range.\
                  \nYou may receive status updates more or less frequently than desired.")
            
        if Errfrac < 0.005 or self.numBins > 36:
            print("Warning: the desired accuracy is very precise, or the number of\
                  \ntoroidal bins is very large. The integration may take a\
                      \nlong time.")
            
        print("Warning: Very fine structure (order of 1 cm^3 or less) may\
              \nnot appear in results, or cause program to take longer to run.")
        print("Warning: Time estimate is probably not accurate.")
        print("integrating RadDist power ...")
        
        # integration function really starts here
        starttime = time.time()
        
        # Function for choosing a random point inside a given rotationally-
        # symmetric volume, where the cross section curve is given as a matplotlib
        # path.
        # Chooses a random point in a rectangular region containing the volume,
        # checks if the point is inside the volume, and repeats until it finds
        # a point that is in the torus.
        def random_uniform_point_noVolume(Wallcurve,\
            Minr, Maxr, Minz, Maxz):
            
            success = 0
            while success == 0:
                x = random.uniform(-Maxr, Maxr)
                y = random.uniform(-Maxr, Maxr)
                z = random.uniform(Minz, Maxz)
                r = np.sqrt((x**2)+(y**2))
                
                if (r < Minr or r > Maxr):
                    pass
                elif Wallcurve.contains_points([(r, z)]):
                    success = 1
                    
            return x, y, z, r
        
        # loading curve
        wallcurve, minr, maxr, minz, maxz =\
            self.tokamak.wallcurve, self.tokamak.minr, self.tokamak.maxr,\
            self.tokamak.minz, self.tokamak.maxz
        
        # define constants and starting values
        angleperbin = 2. * np.pi / self.numBins
        volumeperbin = self.tokamak.volume / float(self.numBins)
        pointsperbin = 0
        reachedprecision = 0
        emissumarray = np.zeros((self.numPuncs, self.numBins))
        emissqsumarray = np.zeros((self.numPuncs, self.numBins))
        
        # calculates the mean and variance of emissivity inside the region
        # by evaluating at randomly chosen points throughout the region
        # and averaging.
        while reachedprecision == 0:
            x, y, z, R\
                = random_uniform_point_noVolume(wallcurve,\
                minr, maxr, minz, maxz)
           
            # finds phi position, rotates to phi position in first bin
            phi = XY_To_RPhi(x,y)[1] # in radians
            if phi < 0:
                phi += 2. * np.pi
            phibin = math.floor(phi / angleperbin)
            phifirstbin = phi - (angleperbin * phibin)
            
            for numbin in range(0, self.numBins):
                phi = phifirstbin + (angleperbin * numbin)
                x, y = RPhi_To_XY(R, phi)
                
                for numpunc in range(1,self.numPuncs+1) :
                    if numpunc == 1:
                        EvalFirstPunc = 1.
                        EvalSecondPunc = 0.
                    elif numpunc == 2:
                        EvalFirstPunc = 0.
                        EvalSecondPunc = 1.
                    # the way this approach handles second punctures may have a problem here; the field line follower gets
                    # called for phi larger than 2 pi (because that's enforced above I guess?). Should check this carefully before second paper
                    emission = self.evaluate(x,y,z, EvalFirstPunc, EvalSecondPunc)
                    emissumarray[numpunc-1, numbin] += emission
                    emissqsumarray[numpunc-1, numbin] += emission**2
    
            pointsperbin += 1
            
            # check if at desired error yet. If yes, show results and end program;
            # otherwise, continue choosing points
            if pointsperbin % Pointsupdate == 0:
                emismeanarray = emissumarray / pointsperbin
                emisvararray = (emissqsumarray / pointsperbin)\
                    - (emismeanarray**2)
                
                integemisarray = volumeperbin * emismeanarray
                totintegemis = np.sum(integemisarray)
                integemisvararray = volumeperbin**2\
                    * emisvararray / pointsperbin
                integemiserrarray = np.sqrt(integemisvararray)
                totintegemiserr = np.sum(integemiserrarray)
                toterrfrac = totintegemiserr / totintegemis
                
                if toterrfrac < Errfrac:
                    reachedprecision = 1
                else:
                    print("total std. err fraction so far = " + str(toterrfrac))
                    timeremainest = 140.0 *\
                        ((toterrfrac / Errfrac) - 1.0) ** (1.0/2.0)
                    #print("random points checked per bin so far = " + str(pointsperbin))
                    runtime = time.time() - starttime
                    print("time so far = " + str(math.ceil(runtime)) + " seconds")
                    try:
                        print("rough estimated time remaining = " + str(math.ceil(timeremainest)) + " seconds")
                    except:
                        print("rough estimated time remaining = NaN seconds")
                    print("")
                    
        # calculations of final radiation values and errors
        emismeanarray = emissumarray / pointsperbin
        emisvararray = (emissqsumarray / pointsperbin)\
            - (emismeanarray**2)
        
        integemisarray = volumeperbin * emismeanarray
        totintegemis = np.sum(integemisarray)
        integemisvararray = volumeperbin**2\
            * emisvararray / pointsperbin
        integemiserrarray = np.sqrt(integemisvararray)
        totintegemiserr = np.sum(integemiserrarray)
        toterrfrac = totintegemiserr / totintegemis
        
        # possible outputs
        #print("bin integrated emissivities = " + str(integemisarray))
        #print("total integrated emissivity = " + str(totintegemis))
        #print("bin integrated emissivity errors = " + str(integemiserrarray))
        #print("total integrated emissivity error = " + str(totintegemiserr))
        #print("total integrated emissivity error fraction = " + str(toterrfrac))
        
        self.powerPerBin = integemisarray.tolist()
        
        runtime = time.time() - starttime
        print("integration runtime = " + str(math.ceil(runtime)) + " seconds")
        print("final error fraction = " + str(toterrfrac))

    def bin_sum(self, Variable_seg):
        variable_save = []
        for arrayIndex in range(len(Variable_seg)):
            channel_sum = []
            for channelIndex in range(len(Variable_seg[arrayIndex])):
                sum = 0
                for binIndex in range(len(Variable_seg[arrayIndex][channelIndex])):
                    sum = sum + Variable_seg[arrayIndex][channelIndex][binIndex]
                channel_sum.append(sum)
            variable_save.append(channel_sum)
        return variable_save

    def bolos_observe(self):

        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass Tokamak = None and Mode = 'Build'\
                  to this RadDist")
            sys.exit(11)
        
        # set random seed to always be the same so that the result is not different every run
        # doesn't seem to work though. some raysect developer comments on github suggest
        # this is a known issue...
        # https://github.com/ray-project/ray/issues/10145
        raysectseed(self.randSeed)
            
        boloCameras = self.tokamak.bolometers
        boloExtras = self.tokamak.extrabolometers
        self.boloCameras_powers = []
        self.boloCameras_powers_2nd = []
        self.boloCameras_powers_extras = []
        self.boloCameras_powers_extras_2nd = []
        if self.torSegmented:
            self.bolo_powers_segmented = []
            self.bolo_powers_segmented_2nd = []
            self.bolo_powers_extras_segmented = []
            self.bolo_powers_extras_segmented_2nd = []
        
        def assign_monte_carlo_iterations_loop(BolosList):
            for channel in BolosList:
                try:
                    for foil in list(channel.bolometer_camera.foil_detectors):
                        # the following line sets the number of rays to trace at once in parallel. Heimdall can apparently do 16.
                        foil.render_engine.processes = 16
                        # The following line sets the resolution of the sightline area viewed by each bolometer. Jack Lovell says this
                        # number is reasonable.
                        foil.pixel_samples = 1000
                except:
                    # this case is for the JET KB5H, 5V, and 1 bolometers which don't have the extra bolometer_camera layer
                    for foil in list(channel.foil_detectors):
                        # the following line sets the number of rays to trace at once in parallel. Heimdall can apparently do 16.
                        foil.render_engine.processes = 16
                        # The following line sets the resolution of the sightline area viewed by each bolometer. Jack Lovell says this
                        # number is reasonable.
                        foil.pixel_samples = 1000

            return BolosList

        for array in boloCameras:
            array.bolometers = assign_monte_carlo_iterations_loop(BolosList=array.bolometers)

        if len(boloExtras) > 0:
            for array in boloExtras:
                array.bolometers = assign_monte_carlo_iterations_loop(BolosList=array.bolometers)
        
        # The following lines define an emitting volume that the RadDist evaluate functions are embedded in.
        # The parameters are set to encompass to entirety of the tokamak (this setup is for JET and smaller), and the -2.5 is to shift the
        # volume so that it is vertically centered at  z = 0.
        # the first line defines the shape, the second makes it an emitting volume, the third embeds it in the world of the tokamak in question
        # whoops I've hardcoded upper and lower height bounds... oh well. Can adjust that if we ever 
        # port this to a machine with larger than these. Use self.tokamak parameters probably
        emitter = Cylinder(radius=5, height=5, transform=translate(0, 0, -2.5))
        emitter.parent = self.tokamak.world

        def arrayLoop(BolosArray, ArrayNumber, PowersArray):
            channel_powers = []
            print("Observing bolometer array #" + str(ArrayNumber))
            for channel in BolosArray.bolometers:
                if self.torSegmented:
                    # not adapted for JET in segmented setup. But JET doesn't have toroidal bolos, so moot
                    bin_powers = []
                    for i in range(self.numBins):
                        self.tempObserverBin = i
                        observeVal = channel.bolometer_camera.observe()
                        if len(observeVal) == 1:
                            observeVal = observeVal[0]
                        bin_powers.append(observeVal)
                    channel_powers.append(bin_powers)
                else:
                    try:
                        observeVal = channel.bolometer_camera.observe()
                    except:
                        # this case is for the JET KB5H, 5V, and 1 bolometers which don't have the extra bolometer_camera layer
                        observeVal = channel.observe()
                    if len(observeVal) == 1:
                        observeVal = observeVal[0]
                    channel_powers.append(observeVal)
            arraynumber = ArrayNumber + 1
            PowersArray.append(channel_powers)

            return arraynumber, PowersArray
        
        def punctureLoop(PunctureNum):
            print("Observing puncture " + str(PunctureNum))
            if PunctureNum==1:
                emitter.material = VolumeTransform(RadiationFunction(\
                    self.evaluate_first_punc_cherab), emitter.transform.inverse())
            elif PunctureNum==2:
                emitter.material = VolumeTransform(RadiationFunction(\
                    self.evaluate_second_punc_cherab), emitter.transform.inverse())
            powers_array = []
            arraynumber = 0
            for array in boloCameras:
                arraynumber, powers_array = arrayLoop(\
                    BolosArray=array, ArrayNumber=arraynumber, PowersArray=powers_array)

            if len(boloExtras) > 0:
                extras_powers_array = []
                for array in boloExtras:
                    arraynumber, extras_powers_array = arrayLoop(\
                        BolosArray=array, ArrayNumber=arraynumber, PowersArray=extras_powers_array)

            if PunctureNum==1:
                if self.torSegmented:
                    self.bolo_powers_segmented = powers_array
                    self.boloCameras_powers = self.bin_sum(Variable_seg=self.bolo_powers_segmented)

                    if len(boloExtras) > 0:
                        self.bolo_powers_extras_segmented = extras_powers_array
                        self.boloCameras_powers_extras = self.bin_sum(Variable_seg=self.bolo_powers_extras_segmented)
                else:
                    self.boloCameras_powers = powers_array
                    if len(boloExtras) > 0:
                        self.boloCameras_powers_extras = extras_powers_array
            elif PunctureNum==2:
                if self.torSegmented:
                    self.bolo_powers_segmented_2nd = powers_array
                    self.boloCameras_powers_2nd = self.bin_sum(Variable_seg=self.bolo_powers_segmented_2nd)

                    if len(boloExtras) > 0:
                        self.bolo_powers_extras_segmented_2nd = extras_powers_array
                        self.boloCameras_powers_extras_2nd = self.bin_sum(Variable_seg=self.bolo_powers_extras_segmented_2nd)
                else:
                    self.boloCameras_powers_2nd = powers_array
                    if len(boloExtras) > 0:
                        self.boloCameras_powers_extras_2nd = extras_powers_array

        punctureLoop(PunctureNum=1)
        if (self.distType == "Helical") or (self.distType == "ElongatedHelical"):
            punctureLoop(PunctureNum=2)

    def plot_in_round_old(self, Title = "Radiation Distribution",\
                 FromWhite = False, Resolution = 60, Alpha = 0.05):
        # Makes a 3d plot of the RadDist. Does not include any toroidal distribution overlay.
        # Radiation magnitude is described by color. Very low value points are removed entirely,
        # giving the general radiation structure shape. Nevertheless, the internal radiation
        # structure is obscured by the external; such is 3d plotting. Adjust resolution and alpha for
        # better viewing.
        
        # has angle of plotting off by pi/2
        
        # The sundial shaped thing shows the bins into which radiated power is separated.
        
        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass\
                  Tokamak = None and Mode = 'Build' to this RadDist")
            sys.exit(1)
            
        # just so the intervals are nice whole-ish numbers if Resolution is even
        resolution = Resolution + 1
        
        # setup for tokamak sundial thing (shows inner first wall radius,
        # outer first wall radius, and bin dividers)
        theta = np.linspace(0, 2 * np.pi, resolution)
        majorRadius = self.tokamak.majorRadius
        minorRadius = self.tokamak.minorRadius
        outerX = (majorRadius + minorRadius) * np.cos(theta)
        outerY = (majorRadius + minorRadius) * np.sin(theta)
        Z = np.zeros(Resolution + 1)
        
        innerX = (majorRadius - minorRadius) * np.cos(theta)
        innerY = (majorRadius - minorRadius) * np.sin(theta)
        
        # each column of these resulting arrays is for one line
        binBorderThetas = np.reshape(np.linspace(0, 2 * np.pi, self.numBins+1), (1,-1))
        borderNums = np.reshape(np.linspace(majorRadius - minorRadius,\
                               majorRadius + minorRadius,\
                                   resolution), (-1, 1))
        borderXs = borderNums @ np.cos(binBorderThetas)
        borderYs = borderNums @ np.sin(binBorderThetas)
        
        # setup points for evaluate function bubble plot
        # xVec actually used for all 3 dimensions
        xVec = np.linspace(-(majorRadius + minorRadius),\
                    majorRadius + minorRadius, resolution)
        evalXs, evalYs, evalZs = np.meshgrid(xVec, xVec, xVec)
        functionVals = np.zeros((resolution, resolution, resolution))
        for xValIndx in range(resolution):
            for yValIndx in range(resolution):
                for zValIndx in range(resolution):
                    functionVals[xValIndx, yValIndx, zValIndx] =\
                        self.evaluate_both_punc(xVec[xValIndx], xVec[yValIndx], xVec[zValIndx])
        functionVals = np.floor(255 * functionVals / np.max(functionVals)).astype(int)
        
        evalXs = np.ravel(evalXs)
        evalYs = np.ravel(evalYs)
        evalZs = np.ravel(evalZs)
        functionVals = np.ravel(functionVals)
        negligible = np.argwhere(functionVals < (0.01 * np.max(functionVals)))
        evalXs = np.delete(evalXs, negligible)
        evalYs = np.delete(evalYs, negligible)
        evalZs = np.delete(evalZs, negligible)
        functionVals = np.delete(functionVals, negligible)
        
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        plotlimit = majorRadius + minorRadius
        ax.set_xlim3d(-plotlimit, plotlimit)
        ax.set_ylim3d(-plotlimit, plotlimit)
        ax.set_zlim3d(-plotlimit, plotlimit)
        # inboard midplane ring
        ax.plot(innerX, innerY, Z, color = '#00ceaa')
        # outboard midplane ring
        ax.plot(outerX, outerY, Z, color = '#00ceaa')
        # bin border lines
        for borderNum in range(self.numBins):
            ax.plot(borderXs[:,borderNum], borderYs[:,borderNum], Z, color = '#00ceaa')
        # evaluate function scatter plot.
        if FromWhite == True:
            ax.scatter(evalXs, evalYs, evalZs, c = plt.cm.Purples(functionVals),
                            alpha = Alpha)
        else:
            ax.scatter(evalXs, evalYs, evalZs, c = plt.cm.plasma(functionVals),
                            alpha = Alpha)
                    
        ax.set_title(Title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        return fig
    
    def plot_in_round(self, Title = None, ShowFirstWall=True,\
                FromWhite = False, Resolution = 10, Alpha = 0.1,\
                PlotSightlines=False, ColoredSightlines=False, RemoveZeroSightlines=True):
        
        # Makes a 3d plot of the RadDist. Does not include any toroidal distribution overlay.
        # Radiation magnitude is described by color. Very low value points are removed entirely,
        # giving the general radiation structure shape. Nevertheless, the internal radiation
        # structure is obscured by the external; such is 3d plotting. Adjust resolution and alpha for
        # better viewing.
        
        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass\
                  Tokamak = None and Mode = 'Build' to this RadDist")
            sys.exit(1)
        
        majorRadius = self.tokamak.majorRadius
        minorRadius = self.tokamak.minorRadius
        
        # just so the intervals are nice whole-ish numbers if Resolution is even
        resolution = Resolution + 1
        
        # setup for rad power bubble plot thing
        
        rRes = math.ceil(Resolution)
        phiRes = math.ceil(3.0*np.pi*Resolution)
        zRes = math.ceil(Resolution*2.2)
        
        evalRs = np.linspace(majorRadius - minorRadius, majorRadius + minorRadius, rRes)
        evalPhis = np.linspace(-np.pi, np.pi, phiRes)
        evalZs = np.linspace(-2.2*minorRadius, 2.2*minorRadius, zRes)
                        
        functionVals = []
        plotZs= []
        plotXs=[]
        plotYs=[]
        
        for evalR in evalRs:
            for evalZ in evalZs:
                if self.tokamak.wallcurve.contains_points([(evalR, evalZ)]):
                    for evalPhi in evalPhis:
                            evalX, evalY = RPhi_To_XY(evalR, evalPhi)
                            
                            functionVal = self.evaluate_both_punc(evalX, evalY, evalZ)
                            plotZs.append(evalZ)
                            plotXs.append(evalX)
                            plotYs.append(evalY)
                            functionVals.append(functionVal)
        
        # convert to RGB scale
        functionMax = np.max(functionVals)
        functionVals = [math.floor(255 * x / functionMax) for x in functionVals]
        functionMax = np.max(functionVals)
        
        # # Delete points where radiated power is less than 1% of maximum
        negligible = np.argwhere(functionVals < (0.01 * functionMax))
        functionVals = np.delete(functionVals, negligible)
        plotZs = np.delete(plotZs, negligible)
        plotXs = np.delete(plotXs, negligible)
        plotYs = np.delete(plotYs, negligible)
        
        # create plot and set bounds
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        plotlimit = majorRadius + minorRadius
        ax.set_xlim3d(-plotlimit, plotlimit)
        ax.set_ylim3d(-plotlimit, plotlimit)
        ax.set_zlim3d(-plotlimit, plotlimit)
        
        # evaluate function scatter plot.
        if FromWhite:
            ax.scatter(plotXs, plotYs, plotZs, c = plt.cm.Purples(functionVals),
                            alpha = Alpha)
        else:
            ax.scatter(plotXs, plotYs, plotZs, c = plt.cm.plasma(functionVals),
                            alpha = Alpha)
            
        # setup for tokamak sundial thing (shows inner first wall radius,
        # outer first wall radius, and bin dividers)
        theta = np.linspace(0, 2 * np.pi, self.numBins+1)
        majorRadius = self.tokamak.majorRadius
        minorRadius = self.tokamak.minorRadius
        outerX = (majorRadius + minorRadius) * np.cos(theta)
        outerY = (majorRadius + minorRadius) * np.sin(theta)
        outerZ = np.zeros(self.numBins+1)
        
        innerX = (majorRadius - minorRadius) * np.cos(theta)
        innerY = (majorRadius - minorRadius) * np.sin(theta)
        innerZ = np.zeros(self.numBins+1)
        
        # each column of these resulting arrays is for one line
        binBorderThetas = np.reshape(np.linspace(0, 2 * np.pi, self.numBins+1), (1,-1))
        borderNums = np.reshape(np.linspace(majorRadius - minorRadius,\
                               majorRadius + minorRadius,\
                                   2), (-1, 1))
        borderXs = borderNums @ np.cos(binBorderThetas)
        borderYs = borderNums @ np.sin(binBorderThetas)
        borderZs = np.zeros(2)
        
        # inboard midplane ring
        ax.plot(innerX, innerY, innerZ, color = '#00ceaa')
        # outboard midplane ring
        ax.plot(outerX, outerY, outerZ, color = '#00ceaa')
        # bin border lines
        for borderNum in range(self.numBins):
            ax.plot(borderXs[:,borderNum], borderYs[:,borderNum], borderZs, color = '#00ceaa')
          
        # For plotting first wall contours
        if ShowFirstWall:
            r = self.tokamak.wallcurve.vertices[:,0]
            z = self.tokamak.wallcurve.vertices[:,1]
            ax.plot3D(r,[0.0]*len(r),z, 'orange')
            ax.plot3D(-1.0*r,[0.0]*len(r),z, 'orange')
            ax.plot3D([0.0]*len(r), r, z, 'orange')
            ax.plot3D([0.0]*len(r), -1.0*r, z, 'orange')
        
        if (Title is not None):
            ax.set_title(Title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0, top=0.9)
        plt.tight_layout()
        
        """
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        """

        if PlotSightlines:
            # this option might only work for SPARC
            bolometers = self.tokamak.bolometers

            if ColoredSightlines:
                boloPowersMax = np.max(self.boloCameras_powers)

            for bolo_camera_indx in range(len(bolometers)):
                bolo_camera = bolometers[bolo_camera_indx]
                for bolo_channel_indx in range(len(bolo_camera.bolometers)):
                    bolo_channel = bolo_camera.bolometers[bolo_channel_indx]
                    for foil in bolo_channel.bolometer_camera:
                        slit_centre = foil.slit.centre_point
                        ax.plot(slit_centre[0], slit_centre[1], slit_centre[2], 'ko')
                        origin, hit, _ = foil.trace_sightline()
                        ax.plot(foil.centre_point[0], foil.centre_point[1], foil.centre_point[2], 'kx')
                        if ColoredSightlines:
                            colorNum =(self.boloCameras_powers[bolo_camera_indx][bolo_channel_indx]\
                                    / boloPowersMax) * 5
                            color = plt.cm.Greys(colorNum)
                            if RemoveZeroSightlines:
                                if colorNum > 0.005:
                                    ax.plot([origin[0], hit[0]], [origin[1], hit[1]], [origin[2], hit[2]], c=color)
                            else:
                                ax.plot([origin[0], hit[0]], [origin[1], hit[1]], [origin[2], hit[2]], c=color)
                        else:
                            color='k'
                            ax.plot([origin[0], hit[0]], [origin[1], hit[1]], [origin[2], hit[2]], c=color)
        
        return fig
    
    def plot_unwrapped(self, TorDistFunc = None,\
        SpotSize = 20, FromWhite = False, Resolution = 20, Alpha = 0.005):
        
        # Makes a 3d plot of the RadDist, unwrapped. Toroidal distribution overlay
        # is only set up for helicals so far, and has not been tested.

        # Radiation magnitude is described by color. Very low value points are removed entirely,
        # giving the general radiation structure shape. Nevertheless, the internal radiation
        # structure is obscured by the external; such is 3d plotting. Adjust resolution and alpha for
        # better viewing.
        
        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass\
                  Tokamak = None and Mode = 'Build' to this RadDist")
            sys.exit(1)
        
        majorRadius = self.tokamak.majorRadius
        minorRadius = self.tokamak.minorRadius
        
        xRes = math.ceil(Resolution*(2.0*minorRadius))
        yRes = math.ceil(Resolution*(2.0*math.pi))
        zRes = math.ceil(Resolution*(4.4*minorRadius))
        
        evalRs = np.linspace(majorRadius - minorRadius, majorRadius + minorRadius, xRes)
        evalPhis = np.linspace(-np.pi, np.pi, yRes)
        evalZs = np.linspace(-2.2*minorRadius, 2.2*minorRadius, zRes)
                        
        functionVals = []
        plotRs = []
        plotPhis = []
        plotZs= []
        
        for evalR in evalRs:
            for evalZ in evalZs:
                if self.tokamak.wallcurve.contains_points([(evalR, evalZ)]):
                    for evalPhi in evalPhis:
                            evalX, evalY = RPhi_To_XY(evalR, evalPhi)
                            
                            if TorDistFunc is None:
                                functionVal = self.evaluate_both_punc(evalX, evalY, evalZ)
                            else:
                                functionValFirst = self.evaluate_first_punc(evalX, evalY, evalZ)
                                functionValSecond = self.evaluate_second_punc(evalX, evalY, evalZ)
                                
                                # Need to reconsider these for 0s vs injector locations...
                                # First Puncture
                                functionValFirst = functionValFirst *\
                                        TorDistFunc(Phi0 = evalPhi)
                                
                                # Second Puncture
                                if evalPhi >= self.tokamak.injectionPhiTor:
                                    functionValSecond = functionValSecond *\
                                        TorDistFunc(Phi0 = evalPhi - (2.0 * math.pi))
                                else:
                                    functionValSecond = functionValSecond *\
                                        TorDistFunc(Phi0 = evalPhi + (2.0 * math.pi))

                                functionVal = functionValFirst + functionValSecond
                                
                            plotRs.append(evalR)
                            plotPhis.append(evalPhi)
                            plotZs.append(evalZ)
                            functionVals.append(functionVal)
        
        # convert to RGB scale
        functionMax = np.max(functionVals) * 1e-1
        functionVals = [math.floor(255 * x / functionMax) for x in functionVals]

        # Delete points where radiated power is less than 1% of maximum
        #negligible = np.argwhere(functionVals < (0.01 * functionMax))
        negligible = np.argwhere(functionVals < np.float64(0.01*255.0))
        functionVals = np.delete(functionVals, negligible) 
        plotRs = np.delete(plotRs, negligible)
        plotPhis = np.delete(plotPhis, negligible)
        plotZs = np.delete(plotZs, negligible)
        
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_xlim3d(majorRadius - minorRadius, majorRadius + minorRadius)
        
        if self.tokamak.tokamakName=="JET":
            ax.set_zlim3d(-1.5 * minorRadius, 2.0 * minorRadius)
            phiLeft = self.tokamak.injectionPhiTor
            phiRight = self.tokamak.injectionPhiTor + (2.0 * np.pi)
                    
        elif self.tokamak.tokamakName=="SPARC":
            ax.set_zlim3d(-2.2 * minorRadius, 2.2 * minorRadius)
            phiLeft = self.tokamak.injectionPhiTor - (2.0 * np.pi)
            phiRight = self.tokamak.injectionPhiTor
            
        elif self.tokamak.tokamakName=="DIIID":
            ax.set_zlim3d(-2.2 * minorRadius, 2.2 * minorRadius)
            phiLeft = self.tokamak.injectionPhiTor - (2.0 * np.pi)
            phiRight = self.tokamak.injectionPhiTor
            
        ax.set_ylim3d(phiLeft, phiRight)
        ax.view_init(elev=20.0, azim=200)
        #ax.view_init(elev=20.0, azim=0)
        
        #Moves center of plot from phi = 0 to injector location
        for distNum in range(len(plotPhis)):
            if plotPhis[distNum] < phiLeft:
                plotPhis[distNum] = plotPhis[distNum] + (2.0 * np.pi)
            elif plotPhis[distNum] > phiRight:
                plotPhis[distNum] = plotPhis[distNum] - (2.0 * np.pi)
           
        # evaluate function scatter plot.
        if FromWhite == True:
            ax.scatter(plotRs, plotPhis, plotZs, c = plt.cm.Purples(functionVals),
                            s=SpotSize, alpha = Alpha)
        else:
            ax.scatter(plotRs, plotPhis, plotZs, c = plt.cm.plasma(functionVals),
                            s=SpotSize, alpha = Alpha)
        
        # For plotting last closed flux surface
        """
        r, z = self.tokamakAMode.get_qsurface_contour(Qval="Separatrix")
        phiEdgeLeft = [phiLeft] * len(r)
        phiEdgeRight = [phiRight] * len(r)
        ax.plot3D(r,phiEdgeLeft,z, 'orange')
        ax.plot3D(r,phiEdgeRight,z, 'orange')
        """
        
        # For plotting first wall contour
        r = self.tokamak.wallcurve.vertices[:,0]
        z = self.tokamak.wallcurve.vertices[:,1]
        phiEdgeLeft = [phiLeft] * len(r)
        phiEdgeRight = [phiRight] * len(r)
        ax.plot3D(r,phiEdgeLeft,z, 'orange')
        ax.plot3D(r,phiEdgeRight,z, 'orange')
        
        ax.set_xlabel("R")
        ax.set_ylabel("$\Phi$")
        ax.set_zlabel("Z")
        ax.grid(False)
        """
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        """
        
        return fig
    
    def plot_crossSec(self, Phi = 0.0, TorDistFunc=None):
        # Makes a 2d plot of the RadDist at a given phi location. Phi is in radians.
        # Does not include any toroidal distribution overlay.
        
        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function, "\
                  + "Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass "\
                  + "Tokamak = None and Mode = 'Build' to this RadDist")
            sys.exit(1)
        
        fig = plt.figure()
        ax = plt.axes()

        if TorDistFunc != None:
            firstPuncTorFactor = TorDistFunc(Phi0=Phi)
            secondPuncClockwiseTorFactor = TorDistFunc(Phi0 = Phi-(2.0*np.pi))
            secondPuncAntiClockwiseTorFactor = TorDistFunc(Phi0 = Phi+(2.0*np.pi))
        
        # make 2d grids of r, z, and emissivity values (all start at 0)
        r = np.linspace(self.tokamak.majorRadius-self.tokamak.minorRadius,\
                        self.tokamak.majorRadius+self.tokamak.minorRadius, num=50)
        z = np.linspace(-2.5*self.tokamak.minorRadius,\
                        2.5*self.tokamak.minorRadius, num=90)
        rgrid,zgrid = np.meshgrid(r,z)
        emis = rgrid*0.
        
        # take a few phi phi values about the desired phi value
        for phi in np.linspace(Phi-0.01, Phi+0.01, 4):
            # convert to x and y
            xgrid, ygrid, = RPhi_To_XY(rgrid, phi)          
            
            for i,j in np.ndindex(xgrid.shape):
                # add emissivity at each point to its point on the emis grid
                x = xgrid[i,j]
                y = ygrid[i,j]
                z1 = zgrid[i,j]
                
                if TorDistFunc == None:
                    emis[i,j] += self.evaluate_both_punc(x,y,z1)
                else:
                    emis[i,j] += self.evaluate_first_punc(x,y,z1) * firstPuncTorFactor
                    
                    if phi >= self.tokamak.injectionPhiTor:
                        emis[i,j] += self.evaluate_second_punc(x,y,z1) *\
                            secondPuncClockwiseTorFactor
                    else:
                        emis[i,j] += self.evaluate_second_punc(x,y,z1) *\
                            secondPuncAntiClockwiseTorFactor

        # make 2d plot
        ax.contourf(r,z, emis, levels=20, cmap="Blues")
        
        # For plotting first wall contour
        r = self.tokamak.wallcurve.vertices[:,0]
        z = self.tokamak.wallcurve.vertices[:,1]
        ax.plot(r,z, 'orange')
        
        ax.set_title("Phi = Injector")
        ax.set_ylabel('Z [m]')
        ax.set_xlabel('R [m]')
        
        ax.set_aspect('equal')
        
        return fig
    
    def observe_local_intensity(self, ApCenterPoint = [2.18, 0.68, 0.0],\
                                FoilRotVec = [0.0, -np.pi/4.0, np.pi]):
        # aperature center point is r,z,phi
        # foil rotation vector is the direction the "foil" is pointing in,
        # the normal vector to the surface where the intensity is observed.
        # first angle is [about the foil?],
        # second angle is angle in x-z plane starting from horizontal outwards,
        # third angle is angle in x-y plane starting from outwards

        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass Tokamak = None and Mode = 'Build'\
                  to this RadDist")
            sys.exit(11)

        # set random seed to always be the same so that the result is not different every run
        # doesn't seem to work though. some raysect developer comments on github suggest
        # this is a known issue...
        # https://github.com/ray-project/ray/issues/10145
        raysectseed(self.randSeed)
            
        tempworld = deepcopy(self.tokamak.world)
        sensor = Synth_Brightness_Observer(World=tempworld,\
                ApCenterPoint = ApCenterPoint,\
                FoilRotVec = FoilRotVec,\
                IndicatorLights=False)
        
        for foil in sensor.bolometer_camera.foil_detectors:
            # the following line sets the number of rays to trace at once in parallel. Heimdall can apparently do 16.
            foil.render_engine.processes = 16
            # The following line sets the resolution of the sightline area viewed by each bolometer. Jack Lovell says this
            # number is reasonable.
            foil.pixel_samples = 1e6 # 1e5
        
        # The following ines define an emitting volume that the RadDist will evaluate functions are embedded in.
        # The parameters are set to encompass to entirety of the tokamak (this setup is for JET and smaller), and the -2.5 is to shift the
        # volume so that it is vertically centered at  z = 0.
        # the first line defines the shape, the second makes it an emitting volume, the third embeds it in the world of the tokamak in question
        emitter = Cylinder(radius=5, height=5, transform=translate(0, 0, -2.5))
        emitter.parent = tempworld
        
        # Observe first punctures
        emitter.material = VolumeTransform(RadiationFunction(\
            self.evaluate_first_punc_cherab), emitter.transform.inverse())
        
        sensor_power_firstpunc = sensor.bolometer_camera.observe()
        
        # Observe second punctures
        emitter.material = VolumeTransform(RadiationFunction(\
            self.evaluate_second_punc_cherab), emitter.transform.inverse())
        
        sensor_power_secondpunc = sensor.bolometer_camera.observe()
        
        #print("Intensity Sensor Power 1st Punc = " + str(sensor_power_firstpunc) + " [W]")
        #print("Intensity Sensor Power 2nd Punc = " + str(sensor_power_secondpunc) + " [W]")

        return(sensor_power_firstpunc[0] + sensor_power_secondpunc[0])
        
class Toroidal(RadDist):
    # this class has a gaussian distribution about a flat toroidal ring
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = None, StartZ = 0.0, PolSigma = 0.15,\
                 SaveFileFolder=None, TorSegmented=False):
        super(Toroidal, self).__init__(NumBins = NumBins,\
                 NumPuncs = 1, Tokamak = Tokamak,\
                 Mode = Mode, LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
        if LoadFileName == None:
            if StartR == None:
                self.startR = self.tokamak.majorRadius
            else:
                self.startR = StartR
            self.startZ = StartZ
            self.polSigma = PolSigma
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.polSigma = properties["polSigma"]
        
        self.distType = "Toroidal"
        
        if Mode == "Build":
            self.make_build_mode()
        
    def make_build_mode(self):
        # toroidals need no additions for build mode
        pass
        
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        
        # first we need to convert from x,y,z to R,Z,phi
        Z = z
        R, phi0 = XY_To_RPhi(x,y)
        
        if EvalFirstPunc:
            # bivariate normal distribution in poloidal plane. 
            # integrated over dR and dZ this function returns 1. 
            localEmis = (1 / (2 * np.pi * self.polSigma**2)
                * math.exp(-0.5 * ((R - self.startR)**2 + (Z - self.startZ)**2) / self.polSigma**2))
            
            return localEmis
            
        if EvalSecondPunc:
            
            return 0.0
        
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self.__dict__
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder,"toroidal_pSig_") + str(round(self.polSigma, RoundDec)).replace('.', '_')\
        + "_R_" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "_Z_" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))
             
class ReflectedToroidal(RadDist):
    # this class has a gaussian distribution about a flat toroidal ring
    # plus a reflection of that gaussian over the xy plane
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = None, StartZ = 0.0, PolSigma = 0.15,\
                 SaveFileFolder=None, TorSegmented=False):
        super(ReflectedToroidal, self).__init__(NumBins = NumBins,\
                 NumPuncs = 1, Tokamak = Tokamak,\
                 Mode = Mode, LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
        if LoadFileName == None:
            if StartR == None:
                self.startR = self.tokamak.majorRadius
            else:
                self.startR = StartR
            self.startZ = StartZ
            self.polSigma = PolSigma
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.polSigma = properties["polSigma"]
        
        self.distType = "ReflectedToroidal"
        
        if Mode == "Build":
            self.make_build_mode()
        
    def make_build_mode(self):
        # toroidals need no additions for build mode
        pass
        
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        
        # first we need to convert from x,y,z to R,Z,phi
        Z = z
        R, phi0 = XY_To_RPhi(x,y)
        
        if EvalFirstPunc:
            # bivariate normal distribution in poloidal plane. 
            # integrated over dR and dZ this function returns 1. 
            localEmis1 = (1 / (2 * np.pi * self.polSigma**2)
                * math.exp(-0.5 * ((R - self.startR)**2 + (Z - self.startZ)**2) / self.polSigma**2))
            
            # add a second gaussian spot refected about xy plane
            localEmis2 = (1 / (2 * np.pi * self.polSigma**2)
                * math.exp(-0.5 * ((R - self.startR)**2 + (Z + self.startZ)**2) / self.polSigma**2))

            localEmis = localEmis1 + localEmis2
            return localEmis
            
        if EvalSecondPunc:
            
            return 0.0
        
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self.__dict__
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder,"reflectedtoroidal_pSig_") + str(round(self.polSigma, RoundDec)).replace('.', '_')\
        + "_R_" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "_Z_" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))

class TiltedReflectedToroidal(RadDist):
    # this class has a gaussian distribution about a flat toroidal ring
    # plus a reflection of that gaussian over the xy plane
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = None, StartZ = 0.0, PolSigma = 0.15,\
                 Elongation = 1.0,\
                 SaveFileFolder=None, TorSegmented=False):
        super(TiltedReflectedToroidal, self).__init__(NumBins = NumBins,\
                 NumPuncs = 1, Tokamak = Tokamak,\
                 Mode = Mode, LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
        if LoadFileName == None:
            if StartR == None:
                self.startR = self.tokamak.majorRadius
            else:
                self.startR = StartR
            self.startZ = StartZ
            self.polSigma = PolSigma
            self.elongation = Elongation
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.polSigma = properties["polSigma"]
            self.elongation = properties["elongation"]
        
        self.distType = "TiltedReflectedToroidal"
        
        if Mode == "Build":
            self.make_build_mode()
        
    def make_build_mode(self):
        vertExtendParam = 3.0 # for vertical extension of plasma... hardcoded for now

        # first we need to decompose (R,Z) in terms of parallel/perpendicular
        # to tilt direction. Approximated as the perpendicular direction
        # to the vector from (major radius, 0) to (startR, startZ)
        # "cent0" = (major radius, 0), "cent1" = (startR, startZ), "point" = (R,Z)
        cent0ToCent1Vec = [self.startR - self.tokamak.majorRadius,self.startZ]
        cent0ToCent1Vec[1] = cent0ToCent1Vec[1] / vertExtendParam
        cent0ToCent1VecMag = math.sqrt(cent0ToCent1Vec[0]**2 + cent0ToCent1Vec[1]**2)
        self.cent0ToCent1VecNormed = [x/cent0ToCent1VecMag\
                            for x in cent0ToCent1Vec]
        self.perpVecNormed = [-self.cent0ToCent1VecNormed[1], self.cent0ToCent1VecNormed[0]]      
        
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        
        # first we need to convert from x,y,z to R,Z,phi
        Z = z
        R, phi0 = XY_To_RPhi(x,y)
        
        if EvalFirstPunc:
            # first (tilted) gaussian spot
            cent1ToPointVec = [R - self.startR, Z - self.startZ]
            paralleldist = cent1ToPointVec[0] * self.cent0ToCent1VecNormed[0] +\
                cent1ToPointVec[1] * self.cent0ToCent1VecNormed[1]
            perpdist = cent1ToPointVec[0] * self.perpVecNormed[0] +\
                cent1ToPointVec[1] * self.perpVecNormed[1]
            
            localEmis1 = ((1.0 / (2.0 * np.pi * self.elongation * (self.polSigma**2)))\
                * math.exp(-0.5 * (perpdist**2) / (self.polSigma*self.elongation)**2)\
                * math.exp(-0.5 * (paralleldist**2) / self.polSigma**2))
            
            # second gaussian spot, reflected over xy plane
            cent1ToPointVec = [R - self.startR, Z + self.startZ]
            paralleldist = cent1ToPointVec[0] * self.cent0ToCent1VecNormed[0] +\
                cent1ToPointVec[1] * (-1.0* self.cent0ToCent1VecNormed[1])
            perpdist = cent1ToPointVec[0] * self.perpVecNormed[0] +\
                cent1ToPointVec[1] * (-1.0 * self.perpVecNormed[1])
            
            localEmis2 = ((1.0 / (2.0 * np.pi * self.elongation * (self.polSigma**2)))\
                * math.exp(-0.5 * (perpdist**2) / (self.polSigma*self.elongation)**2)\
                * math.exp(-0.5 * (paralleldist**2) / self.polSigma**2))

            localEmis = localEmis1 + localEmis2
            return localEmis
            
        if EvalSecondPunc:
            
            return 0.0
        
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self.__dict__
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder,"tiltedreflectedtoroidal_pSig_") + str(round(self.polSigma, RoundDec)).replace('.', '_')\
        + "_R_" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "_Z_" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + "_e_" + str(self.elongation).replace('.', '_')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))

class ElongatedRing(RadDist):
    # this class has a bivariate gaussian distribution about a circle, with different
    # gaussian widths in r and z
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = None, StartZ = 0.0, PolSigma = 0.15,\
                 Elongation=1.0, SaveFileFolder=None, TorSegmented=False):
        super(ElongatedRing, self).__init__(NumBins = NumBins,\
                 NumPuncs = 1, Tokamak = Tokamak,\
                 Mode = Mode, LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
        if LoadFileName == None:
            if StartR == None:
                self.startR = self.tokamak.majorRadius
            else:
                self.startR = StartR
            self.startZ = StartZ
            self.polSigma = PolSigma
            self.elongation = Elongation
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.polSigma = properties["polSigma"]
            self.elongation = properties["elongation"]
        
        self.distType = "ElongatedRing"
        
        if Mode == "Build":
            self.make_build_mode()
        
    def make_build_mode(self):
        # toroidally symmetric radDists need no additions for build mode
        pass
        
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        
        # first we need to convert from x,y,z to R,Z,phi
        Z = z
        R, phi0 = XY_To_RPhi(x,y)
        
        if EvalFirstPunc:
            # bivariate normal distribution in poloidal plane. 
            # integrated over dR and dZ this function returns 1. I think.
            localEmis = ((1.0 / (2.0 * np.pi * self.elongation * self.polSigma**2))\
                * math.exp(-0.5 * ((R - self.startR)**2) / self.polSigma**2)\
                * math.exp(-0.5 * ((Z - self.startZ)**2) / (self.polSigma*self.elongation)**2))
            
            return localEmis
            
        if EvalSecondPunc:
            
            return 0.0
        
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self.__dict__
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder,"eRing_pSig_") + str(round(self.polSigma, RoundDec)).replace('.', '_')\
        + "_elong_" + str(round(self.elongation, RoundDec)).replace('.', '_')\
        + "_R_" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "_Z_" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))
                 
class Helical(RadDist):
    # this class will have specifically helical radiation distributions
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = 2.96, StartZ = 0.0, PolSigma = 0.15,\
                 ShotNumber = None, Time=0, SaveFileFolder=None, TorSegmented=False):
        super(Helical, self).__init__(NumBins = NumBins, NumPuncs = 2,\
                 Tokamak = Tokamak, Mode = Mode,\
                 LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
    
        if LoadFileName == None:
            self.startR = StartR
            self.startZ = StartZ
            self.shotnumber = ShotNumber
            self.polSigma = PolSigma
            self.time = Time
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.shotnumber = properties["shotnumber"]
            self.polSigma = properties["polSigma"]
            self.time = properties["time"]
            
        self.distType = "Helical"
        
        if Mode == "Build":
            self.make_build_mode()
            
    def make_build_mode(self):
        self.tokamak.set_fieldline(StartR = self.startR, StartZ = self.startZ,\
            StartPhi = self.tokamak.injectionPhiTor, FlineShotNumber = self.shotnumber)
    
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        # Return the emissivity (W/m^3/rad) at the point (x,y,z) according to this
        # instantiation of Emis3D. This function works only with scalar values of x, y and z.

        # first we need to convert from x,y,z to R,Z,phi
        Z = z
        R, phi0 = XY_To_RPhi(x,y)
        # readjust range of phi to center on injector location
        if phi0 <= self.tokamak.injectionPhiTor - np.pi:
            phi0 = phi0 + 2.*np.pi

        localEmis0 = 0.
        localEmis1 = 0.
        if EvalFirstPunc:
            # next we need the R,Z position of our helical structure at this phi
            flR, flZ = self.tokamak.find_RZ_Fline(Phi=phi0)

            # bivariate normal distribution in poloidal plane. 
            # integrated over dR and dZ this function returns 1. 
            localEmis0 = (1 / (2 * np.pi * self.polSigma**2)
                         * math.exp(-0.5 * ((R - flR)**2 + (Z - flZ)**2) / self.polSigma**2))

        if EvalSecondPunc:
            # this is the setup for 2 times around
            if phi0 >= self.tokamak.injectionPhiTor:
                phi1 = phi0 - 2.*np.pi
            else:
                phi1 = phi0 + 2.*np.pi   

            # next we need the R,Z position of our helical structure at this phi
            flR, flZ = self.tokamak.find_RZ_Fline(Phi=phi1)

            # bivariate normal distribution in poloidal plane. 
            # integrated over dR and dZ this function returns 1. 
            localEmis1 = (1 / (2 * np.pi * self.polSigma**2)
                          * math.exp(-0.5 * ((R - flR)**2 + (Z - flZ)**2) / self.polSigma**2))

        return localEmis0 + localEmis1
        
        
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self.__dict__.copy()
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder ,"helical_shot_") + str(self.shotnumber)\
        + "_pSig_" + str(round(self.polSigma, RoundDec)).replace('.', '_')\
        + "_t_" + str(round(self.time, RoundDec)).replace('.', '_')\
        + "_R_" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "_Z_" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))

class ElongatedHelical(RadDist):
    # this class will have specifically helical radiation distributions
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = 2.96, StartZ = 0.0, PolSigma = 0.15,\
                 Elongation=1.0, Zoffset = 0.0,\
                 ShotNumber = None, Time=0, SaveFileFolder=None, TorSegmented=False):
        super(ElongatedHelical, self).__init__(NumBins = NumBins, NumPuncs = 2,\
                 Tokamak = Tokamak, Mode = Mode,\
                 LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
    
        if LoadFileName == None:
            self.startR = StartR
            self.startZ = StartZ
            self.shotnumber = ShotNumber
            self.polSigma = PolSigma
            self.time = Time
            self.elongation = Elongation
            self.zoffset = Zoffset
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.shotnumber = properties["shotnumber"]
            self.polSigma = properties["polSigma"]
            self.time = properties["time"]
            self.elongation = properties["elongation"]
            if "zoffset" in properties:
                self.zoffset = properties["zoffset"]
            else:
                self.zoffset = 0.0
            
        self.distType = "ElongatedHelical"
        
        if Mode == "Build":
            self.make_build_mode()
            
    def make_build_mode(self):
        self.tokamak.set_fieldline(StartR = self.startR, StartZ = self.startZ,\
            StartPhi = self.tokamak.injectionPhiTor, FlineShotNumber = self.shotnumber)
    
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        # Return the emissivity (W/m^3/rad) at the point (x,y,z) according to this
        # instantiation of Emis3D. This function works only with scalar values of x, y and z.

        # first we need to convert from x,y,z to R,Z,phi
        Z = z
        R, phi0 = XY_To_RPhi(x,y)
        # readjust range of phi to center on injector location
        if phi0 <= self.tokamak.injectionPhiTor - np.pi:
            phi0 = phi0 + 2.*np.pi

        localEmis0 = 0.0
        localEmis1 = 0.0
        vertExtendParam = 3.0 # for vertical extension of plasma... hardcoded for now
        if EvalFirstPunc:
            # next we need the R,Z position of our helical structure at this phi
            flR, flZ = self.tokamak.find_RZ_Fline(Phi=phi0)

            # now for bivariate normal distribution in poloidal plane. 
            # elongated in approximate poloidal direction of field line

            # first we need to decompose (R,Z) in terms of parallel/perpendicular
            # to approximate field line. Approximated as the perpendicular direction
            # to the vector from (major radius, zoffset) to (flR, flZ)
            # "cent0" = (major radius, zoffset), "cent1" = (flR, flZ), "point" = (R,Z)
            cent0ToCent1Vec = [flR - self.tokamak.majorRadius, flZ - self.zoffset]
            cent0ToCent1Vec[1] = cent0ToCent1Vec[1] / vertExtendParam
            cent0ToCent1VecMag = math.sqrt(cent0ToCent1Vec[0]**2 + cent0ToCent1Vec[1]**2)
            cent0ToCent1VecNormed = [x/cent0ToCent1VecMag\
                              for x in cent0ToCent1Vec]
            perpVecNormed = [-cent0ToCent1VecNormed[1], cent0ToCent1VecNormed[0]]
            cent1ToPointVec = [R - flR, Z - flZ]
            paralleldist = cent1ToPointVec[0] * cent0ToCent1VecNormed[0] +\
                cent1ToPointVec[1] * cent0ToCent1VecNormed[1]
            perpdist = cent1ToPointVec[0] * perpVecNormed[0] +\
                cent1ToPointVec[1] * perpVecNormed[1]
            
            localEmis0 = ((1.0 / (2.0 * np.pi * self.elongation * (self.polSigma**2)))\
                * math.exp(-0.5 * (perpdist**2) / (self.polSigma*self.elongation)**2)\
                * math.exp(-0.5 * (paralleldist**2) / self.polSigma**2))

        if EvalSecondPunc:
            # this is the setup for 2 times around
            if phi0 >= self.tokamak.injectionPhiTor:
                phi1 = phi0 - 2.*np.pi
            else:
                phi1 = phi0 + 2.*np.pi   

            # next we need the R,Z position of our helical structure at this phi
            flR, flZ = self.tokamak.find_RZ_Fline(Phi=phi1)

            # now for bivariate normal distribution in poloidal plane. 
            # elongated in approximate poloidal direction of field line

            # first we need to decompose (R,Z) in terms of parallel/perpendicular
            # to approximate field line. Approximated as the perpendicular direction
            # to the vector from (major radius, zoffset) to (flR, flZ)
            # "cent0" = (major radius, zoffset), "cent1" = (flR, flZ), "point" = (R,Z)
            cent0ToCent1Vec = [flR - self.tokamak.majorRadius, flZ - self.zoffset]
            cent0ToCent1Vec[1] = cent0ToCent1Vec[1] / vertExtendParam
            cent0ToCent1VecMag = math.sqrt(cent0ToCent1Vec[0]**2 + cent0ToCent1Vec[1]**2)
            cent0ToCent1VecNormed = [x/cent0ToCent1VecMag\
                              for x in cent0ToCent1Vec]
            perpVecNormed = [-cent0ToCent1VecNormed[1], cent0ToCent1VecNormed[0]]
            cent1ToPointVec = [R - flR, Z - flZ]
            paralleldist = cent1ToPointVec[0] * cent0ToCent1VecNormed[0] +\
                cent1ToPointVec[1] * cent0ToCent1VecNormed[1]
            perpdist = cent1ToPointVec[0] * perpVecNormed[0] +\
                cent1ToPointVec[1] * perpVecNormed[1]
            
            localEmis1 = ((1.0 / (2.0 * np.pi * self.elongation * self.polSigma**2))\
                * math.exp(-0.5 * (perpdist**2) / (self.polSigma*self.elongation)**2)\
                * math.exp(-0.5 * (paralleldist**2) / self.polSigma**2))

        return localEmis0 + localEmis1
            
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self.__dict__.copy()
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder ,"ehelical_shot_") + str(self.shotnumber)\
        + "_pSig_" + str(round(self.polSigma, RoundDec)).replace('.', '_')\
        + "_t_" + str(round(self.time, RoundDec)).replace('.', '_')\
        + "_R_" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "_Z_" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + "_e_" + str(self.elongation).replace('.', '_')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))

class Sphere(RadDist):
    # these structures are uniform spheres of emissivity
    def __init__(self, NumBins = 18, Tokamak = None,\
               Mode = "Analysis", LoadFileName = None,\
               StartR = 1.2, StartZ = 0.0, Radius = .5,\
               Phi = 0, SaveFileFolder=None, TorSegmented=False):
        super(Sphere, self).__init__(NumBins = NumBins,\
             NumPuncs = 1, Tokamak = Tokamak,\
             Mode = Mode, LoadFileName = LoadFileName,\
             SaveFileFolder=SaveFileFolder, TorSegmented=TorSegmented)
        
        if LoadFileName == None:
            if StartR == None:
                self.startR = self.tokamak.majorRadius
            else:
                self.startR = StartR
            self.startZ = StartZ
            self.radius = Radius
            self.phi = Phi
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.startR = properties["startR"]
            self.startZ = properties["startZ"]
            self.radius = properties["radius"]
            self.phi = properties["phi"]
        
        self.distType = "Sphere"
        
    def make_build_mode(self):
        # toroidals need no additions for build mode
        pass
        
   
    def evaluate(self, x,y,z, EvalFirstPunc, EvalSecondPunc):
        
        # first we need to convert from x,y,z to R,Z,phi
        
        
        startX, startY = RPhi_To_XY(self.startR, self.phi)
        

        if EvalFirstPunc:
            #Any point outside of major radius of Tokamak has no emisivisty  
            if((x-startX)**2 + (y-startY)**2 + (z-self.startZ)**2 < self.radius**2):
                localEmis = 1
            else: 
                localEmis = 0
            return localEmis
            
        if EvalSecondPunc:
            
            return 0.0
        
    def save_RadDist(self, RoundDec = 2):
        #[saves radiation distribution to a file]
        
        properties = self._dict_
        properties = self.prepare_for_JSON(properties)
        
        saveFileName = join(self.saveFileFolder,"sphere_radius_") + str(round(self.radius, RoundDec)).replace('.', '_')\
        + "phi" + str(round(self.phi, RoundDec)).replace('.', '_')\
        + "R" + str(round(self.startR, RoundDec)).replace('.', '_')\
        + "Z" + str(round(self.startZ, RoundDec)).replace('.', '_').replace('-', 'neg')\
        + ".txt"
          
        with open(saveFileName, 'w') as save_file:
             save_file.write(json.dumps(properties))