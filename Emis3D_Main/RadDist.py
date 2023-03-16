# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:12:06 2021

@author: bemst
"""
import numpy as np
import math
import random
import time
import sys
from os.path import dirname, realpath, join
import json
from Util import XY_To_RPhi, RPhi_To_XY

# raysect dependencies
from raysect.core.math import translate
from raysect.optical.material import VolumeTransform
from raysect.primitive import Cylinder

from cherab.tools.emitters import RadiationFunction

class RadDist(object):
    
    def __init__(self, NumBins = 18, NumPuncs = 2, Tokamak = None,\
                 Mode = " Analysis", LoadFileName = None, SaveFileFolder=None):
        #[creates radiation distribution from a given evaluate function]
        
        if LoadFileName == None:
            self.numBins = NumBins
            self.numPuncs = NumPuncs
            self.powerPerBin = []
            self.boloCameras_powers = []
            self.boloCameras_powers_2nd = []
        else:
            with open(LoadFileName) as file:
                properties = json.load(file)
            self.numBins = properties["numBins"]
            self.numPuncs = properties["numPuncs"]
            self.boloCameras_powers = properties["boloCameras_powers"]
            self.boloCameras_powers_2nd = properties["boloCameras_powers_2nd"]
            self.powerPerBin = properties["powerPerBin"]
            
        self.tokamak = Tokamak
        self.saveFileFolder=SaveFileFolder

    
    def prepare_for_JSON(self, properties):
        
        # JSON doesn't know how to deal with objects; just save names to
        # rebuild if necessary
        properties["tokamakName"] = self.tokamak.tokamakName
        properties.pop("tokamak")
        
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
            return self.evaluate(X,Y,Z, EvalFirstPunc=1, EvalSecondPunc=0)
    
    def evaluate_second_punc(self, X,Y,Z):
            return self.evaluate(X,Y,Z, EvalFirstPunc=0, EvalSecondPunc=1)
        
    # Various tokamaks use different toroidal angle conventions than Cherab. Emis3D uses the Cherab
    # Angle convention. self.tokamak.torConventionPhi accounts for this difference
    def evaluate_first_punc_cherab(self, X,Y,Z):
        
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
    
    # Calculates total radiated power per solid angle emitted inside
    # a toroidal region with emissivity function from evaluate function
    def power_per_bin_calc(self, Errfrac = 0.01, Pointsupdate = int(1e5)):
    
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
        
    def bolos_observe(self):

        if self.tokamak.mode != "Build":
            print("The tokamak object of this RadDist is not in Build mode. To use this function,\
                  Either pass Tokamak = [a tokamak object with mode == 'Build'], or pass Tokamak = None and Mode = 'Build'\
                  to this RadDist")
            sys.exit(11)
            
        boloCameras = self.tokamak.bolometers
        self.boloCameras_powers = []
        self.boloCameras_powers_2nd = []
        
        for array in boloCameras:
            for channel in array.bolometers:
                for foil in list(channel.bolometer_camera.foil_detectors):
                    # the following line sets the number of rays to trace at once in parallel. Heimdall can apparently do 16.
                    foil.render_engine.processes = 16
                    # The following line sets the resolution of the sightline area viewed by each bolometer. Jack Lovell says this
                    # number is reasonable.
                    foil.pixel_samples = 1000
        
        # The following ines define an emitting volume that the RadDist will evaluate functions are embedded in.
        # The parameters are set to encompass to entirety of the tokamak (this setup is for JET and smaller), and the -2.5 is to shift the
        # volume so that it is vertically centered at  z = 0.
        # the first line defines the shape, the second makes it an emitting volume, the third embeds it in the world of the tokamak in question
        emitter = Cylinder(radius=5, height=5, transform=translate(0, 0, -2.5))
        emitter.parent = self.tokamak.world
        
        # Observe first punctures
        emitter.material = VolumeTransform(RadiationFunction(\
            self.evaluate_first_punc_cherab), emitter.transform.inverse())
        arraynumber = 0
        for array in boloCameras:
            array_powers = []
            print("Observing bolometer array #" + str(arraynumber))
            arraynumber = arraynumber + 1
            for channel in array.bolometers:
                observeVal = channel.bolometer_camera.observe()
                if len(observeVal) == 1:
                    observeVal = observeVal[0]
                array_powers.append(observeVal)
            self.boloCameras_powers.append(array_powers)
        
        # Observe second punctures
        emitter.material = VolumeTransform(RadiationFunction(\
            self.evaluate_second_punc_cherab), emitter.transform.inverse())
        arraynumber = 0
        for array in boloCameras:
            array_powers = []
            print("Observing bolometer array #" + str(arraynumber) + " second puncture")
            arraynumber = arraynumber + 1
            for channel in array.bolometers:
                if self.distType == "Helical":
                    observeVal = channel.bolometer_camera.observe()
                    if len(observeVal) == 1:
                        observeVal = observeVal[0]
                else:
                    observeVal = 0.0
                array_powers.append(observeVal)
            self.boloCameras_powers_2nd.append(array_powers)

class Toroidal(RadDist):
    # this class has a gaussian distribution about a flat toroidal ring
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = None, StartZ = 0.0, PolSigma = 0.15,\
                 SaveFileFolder=None):
        super(Toroidal, self).__init__(NumBins = NumBins,\
                 NumPuncs = 1, Tokamak = Tokamak,\
                 Mode = Mode, LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder)
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
             
             
class Helical(RadDist):
    # this class will have specifically helical radiation distributions
    def __init__(self, NumBins = 18, Tokamak = None,\
                 Mode = "Analysis", LoadFileName = None,\
                 StartR = 2.96, StartZ = 0.0, PolSigma = 0.15,\
                 ShotNumber = None, Time=0, SaveFileFolder=None):
        super(Helical, self).__init__(NumBins = NumBins, NumPuncs = 2,\
                 Tokamak = Tokamak, Mode = Mode,\
                 LoadFileName = LoadFileName,\
                 SaveFileFolder=SaveFileFolder)
    
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
            self.tokamak.set_fieldline(StartR = self.startR, StartZ = self.startZ,\
                                       StartPhi = self.tokamak.injectionPhiTor)
        
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