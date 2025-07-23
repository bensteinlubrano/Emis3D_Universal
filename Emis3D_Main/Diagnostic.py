#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu
"""

from os.path import join, dirname, realpath

import numpy as np

from raysect.core import Point3D, Vector3D, Node, rotate_basis, translate
from raysect.primitive import Box, Subtract, Sphere
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface, UniformSurfaceEmitter
from raysect.optical.library.spectra.colours import red, green, blue, purple

try:
    from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil
except:
    # Some versions of cherab these objects are inside a file called bolometry, some not... This version is jack lovell's Cherab-stable on JET
    from cherab.tools.observers.bolometry import BolometerCamera, BolometerSlit, BolometerFoil

from scipy.spatial.transform import Rotation as Rot
import configparser

FILE_PATH = dirname(realpath(__file__))
EMIS3D_PARENT_DIRECTORY = dirname(FILE_PATH)
EMIS3D_INPUTS_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "Emis3D_Inputs")

class Diagnostic(object):

    def __init__(self, World=World()):
        self.world = World
        self.name = None

class Bolometer_Camera(Diagnostic):
    '''
    Child class of Diagnostic. This is a bolometer camera. 
    '''
        
    def __init__(self, BoloConfigFiles, World, IndicatorLights=True):
        super(Bolometer_Camera,self).__init__(World=World)
        
        self.bolometers = []
        self.indicatorLights = IndicatorLights
        
            
class Cherab_Basic_Bolo_Camera(Bolometer_Camera):
    # Child class of Bolometer_Camera

    def __init__(self, BoloConfigFiles, World, IndicatorLights):
        super(Cherab_Basic_Bolo_Camera,self).__init__(BoloConfigFiles=BoloConfigFiles, World=World,\
             IndicatorLights=IndicatorLights)
        
        self.build(BoloConfigFiles = BoloConfigFiles)
        
    def build(self, BoloConfigFiles):
        for fileNum in range(len(BoloConfigFiles)):
            newBolometer = CherabBasicBolometer(BoloConfigFile = BoloConfigFiles[fileNum], World=self.world,\
                IndicatorLights = self.indicatorLights)
            self.bolometers.append(newBolometer)

class Synth_Brightness_Observer(Diagnostic):
    # a simple synthetic foil for finding local brightness, e.g. on a first wall tile.
    # less complicated than bolometer camera. No multiple channel options, not designed
    # for any casing. Some parameters like size currently hardcoded
    
    def __init__(self, World, ApCenterPoint, FoilRotVec, IndicatorLights):
        self.world = World
        self.indicatorLights = IndicatorLights
        
        # sets 1cm^2 foil
        self.det_xi = 1e-2 # detector width (meters)
        self.det_zeta = 1e-2  # detector height (meters)
        
        # sets circular aperture with 10cm diameter. "width" is misleading here:
        # bolometer apertures are circular in cherab, and then cut down to
        # rectangles by introducing casing geometry. No casing used here, so
        # aperture is just circular. Only the maximum of these two parameters
        # matters
        self.ap_y = 1e-1 # aperture width (meters)
        self.ap_z = 1e-1 # aperture width (meters)
        
        # sets foil as one millimeter behind 'aperture'
        self.x0 = np.array([-1e-3, 0.0, 0.0]) # detector coordinate system (center) (meters)
        self.x1 = np.array([-1e-3, 0.0, 1.0]) # detector coordinate system (zeta) (meters)
        self.x2 = np.array([-1e-3, 1.0, 0.0]) # detector coordinate system (xi) (meters)
        
        # sets center point of aperture as ApCenterPoint
        self.ap_ro = ApCenterPoint[0] # major radius of aperture (m)
        self.ap_zo = ApCenterPoint[1]  # height of aperture (m)
        self.ap_phi = ApCenterPoint[2]
        
        # sets vector from foil center to aperture center as FoilNormVec
        self.ap_alpha = FoilRotVec[0] # aperture rotation angle alpha (radians)
        self.ap_beta = FoilRotVec[1] # aperture rotation angle beta (radians)
        self.ap_gamma = FoilRotVec[2]  # aperture rotation angle gamma (radians)
        
        self.build()
        
        if self.indicatorLights:
            self.add_indicator_lights()
        
    def build(self):

        # Convenient constants
        # XAXIS = Vector3D(1, 0, 0) #unused
        YAXIS = Vector3D(0, 1, 0)
        ZAXIS = Vector3D(0, 0, 1)
        ORIGIN = Point3D(0, 0, 0)

        xivec = (self.x2 - self.x0)/np.linalg.norm(self.x2 - self.x0)
        zetavec = (self.x1 - self.x0)/np.linalg.norm(self.x1 - self.x0)
        xivec = Vector3D(xivec[0], xivec[1], xivec[2])
        zetavec = Vector3D(zetavec[0], zetavec[1], zetavec[2])
        
        # Bolometer geometry
        SLIT_WIDTH = self.ap_y
        SLIT_HEIGHT = self.ap_z
        FOIL_WIDTH = self.det_xi
        FOIL_HEIGHT = self.det_zeta
        FOIL_CORNER_CURVATURE = 0.0005

        world = self.world

        ########################################################################
        # Build a bolometer camera
        ########################################################################

        # Instance of the bolometer camera
        bolometer_camera = BolometerCamera(camera_geometry=None, parent=world)
        
        # The bolometer slit in this instance just contains targeting information
        # for the ray tracing, since there is no casing geometry
        # The slit is defined in the local coordinate system of the camera
        slit = BolometerSlit(centre_point=ORIGIN, slit_id="sensor slit",\
                             basis_x=YAXIS, dx=SLIT_WIDTH, basis_y=ZAXIS, dy=SLIT_HEIGHT,\
                             parent=bolometer_camera)

        foil_transform = translate(self.x0[0], self.x0[1], self.x0[2])

        foil = BolometerFoil(detector_id="Foil {}".format(1),
                             centre_point=ORIGIN.transform(foil_transform),
                             basis_x=xivec, dx=FOIL_WIDTH,
                             basis_y=zetavec, dy=FOIL_HEIGHT,
                             slit=slit, parent=bolometer_camera, units="Power",
                             accumulate=False, curvature_radius=FOIL_CORNER_CURVATURE)
        
        bolometer_camera.add_foil_detector(foil)


        # the last step here is to translate from alpha, beta, gamma rotations to a single Affine matrix.
        # we know that we can get the correct viewing geometry by setting the forward direction and the up
        # direction such that the cross-product of up x forward is in the desired direction. Recall that the
        # rotate_basis function is defined like rotate_basis(forward, up). So, we will start with a vector in
        # the +x direction (our forward) and in the +z direction (our up), rotate it with alpha beta and gamma,
        # and then find the Cherab up x forward that gives the same result.
        ourForward = np.array([1,0,0])
        ourUp = np.array([0,0,1])
        alphaMat = Rot.from_euler('x', self.ap_alpha)
        alphaMat = alphaMat.as_matrix()
        betaMat = Rot.from_euler('y', -self.ap_beta)
        betaMat = betaMat.as_matrix()
        gammaMat = Rot.from_euler('z', self.ap_gamma)
        gammaMat = gammaMat.as_matrix()

        # rotating the forward vector
        ourForwardR = np.matmul(alphaMat,ourForward)
        ourForwardR = np.matmul(betaMat,ourForwardR)
        ourForwardR = np.matmul(gammaMat,ourForwardR)

        # rotating the up vector
        ourUpR = np.matmul(alphaMat, ourUp)
        ourUpR = np.matmul(betaMat, ourUpR)
        ourUpR = np.matmul(gammaMat, ourUpR)

        # ourForwardR is now the direction we would like to look. that means CherabUp x CherabForward = ourForwardR.
        # not enough constraints yet, we also need to use ourUpR. Last constraint is that ourUpR should be
        # identically CherabForward. So we have CherabUp x ourUpR = ourForwardR.
        cherabUp = np.cross(ourUpR, ourForwardR, axis=None)
        ourUpR = Vector3D(ourUpR[0], ourUpR[1], ourUpR[2])
        cherabForward = ourUpR
        cherabUp = Vector3D(cherabUp[0], cherabUp[1], cherabUp[2])

        # we've got the camera properly aligned, but we are not translating properly. we need to rotate the R Z translation
        # to get the proper XYZ.
        transVec = np.array([self.ap_ro, 0., 0.])
        gammaMatTrans = Rot.from_euler('z', self.ap_phi)
        gammaMatTrans = gammaMatTrans.as_matrix()
        transVec = np.matmul(gammaMatTrans, transVec)
        
        bolometer_camera.transform = translate(transVec[0],transVec[1],self.ap_zo)\
            *rotate_basis(cherabForward, cherabUp) 

        
        self.bolometer_camera = bolometer_camera
        
    def add_indicator_lights(self, Size=0.002, NormVecSize = 0.02):        
        # Adds lights indicating center points of foil and the normal vectors to it
        foil = self.bolometer_camera.foil_detectors[0]
        center_point = foil.centre_point
        print(center_point)
        CenterPoint = Sphere(Size, parent=self.world,\
            transform=translate(center_point.x, center_point.y, center_point.z))
        CenterPoint.material = UniformSurfaceEmitter(red, 100.0)
        print("in")
        normal_vector = foil.normal_vector * NormVecSize
        print(normal_vector)
        print("")
        NormalPoint = Sphere(Size, parent=self.world,\
            transform=translate(center_point.x+normal_vector.x, center_point.y+normal_vector.y, center_point.z+normal_vector.z))
        NormalPoint.material = UniformSurfaceEmitter(green, 100.0)
            
        # Adds lights indicating center points of slits
        for i in range(len(self.bolometer_camera.slits)):
            slit_center_point = self.bolometer_camera.slits[i].centre_point
            print(slit_center_point)
            SlitPoint = Sphere(Size, parent=self.world,\
                transform=translate(slit_center_point.x, slit_center_point.y, slit_center_point.z))
            SlitPoint.material = UniformSurfaceEmitter(blue, 100.0)
            
            normal_vector = self.bolometer_camera.slits[i].normal_vector * NormVecSize
            print(normal_vector)
            print("")
            NormalPoint = Sphere(Size, parent=self.world,\
                transform=translate(slit_center_point.x+normal_vector.x, slit_center_point.y+normal_vector.y, slit_center_point.z+normal_vector.z))
            NormalPoint.material = UniformSurfaceEmitter(purple, 100.0)
        ## end of testing loops    

class Bolometer(object):
    '''
    Individual Bolometers (or sets of bolometers sharing an aperature) within a bolometer camera
    '''
    def __init__(self, BoloConfigFile, World, IndicatorLights):
        self.world = World
        self.boloConfigFile = self.reformat_Bolo_Config_File(BoloConfigFile) # filename of configuration    
        self.indicatorLights=IndicatorLights
        self.dtype = None # diagnostic type
        self.notes = None
        self.ap_ro =None  # major radius of aperture (m)
        self.ap_zo =None  # height of aperture (m)
        self.ap_alpha =None  # aperture rotation angle alpha (radians)
        self.ap_beta = None # aperture rotation angle beta (radians)
        self.ap_gamma =None  # aperture rotation angle gamma (radians)
        self.ap_y = None # aperture width (meters)
        self.ap_z =None  # aperture width (meters)
        self.x0 =None  # detector coordinate system (center) (meters)
        self.x1 =None  # detector coordinate system (zeta) (meters)
        self.x2 =None  # detector coordinate system (xi) (meters)
        self.det_xi = None # detector width (meters)
        self.det_zeta =None  # detector height (meters)
        self.xi =None  # xi locations of detector centers (meters)
        self.zeta =None  # zeta locations of detector centers (meters)
        self.xi_ch =None  # xi spacing of channels (meters)
        self.n_xi = None # number of xi channels
        self.xi_o =None  # xi location of center channel
        self.zeta_ch = None # zeta spacing of channels (meters)
        self.n_zeta =None # number of zeta channels
        self.zeta_o =None  # zeta location of center channel
        self.etendues = None # etendue
        self.etendue_errors = None
        self.read_Bolo_Config_File()
        
            
    def calc_etendues(self):
        
        raytraced_etendues = []
        raytraced_errors = []
        analytic_etendues = []
        for foil in self.bolometer_camera:
            raytraced_etendue, raytraced_error = foil.calculate_etendue(ray_count=400000, max_distance=2.0 * abs(self.x0[0]))
            Adet = foil.x_width * foil.y_width
            Aslit = foil.slit.dx * foil.slit.dy
            costhetadet = foil.sightline_vector.normalise().dot(foil.normal_vector)
            costhetaslit = foil.sightline_vector.normalise().dot(foil.slit.normal_vector)
            distance = foil.centre_point.vector_to(foil.slit.centre_point).length
            analytic_etendue = Adet * Aslit * costhetadet * costhetaslit / distance**2
            print("{} raytraced etendue: {:.4g} +- {:.1g} analytic: {:.4g}".format(
                foil.name, raytraced_etendue, raytraced_error, analytic_etendue))
            raytraced_etendues.append(raytraced_etendue)
            raytraced_errors.append(raytraced_error)
            analytic_etendues.append(analytic_etendue)
        self.etendues = raytraced_etendues
        self.etendue_errors = raytraced_errors
        
    def add_etendue_configfile(self, BoloConfigFile):
        filer = open(BoloConfigFile, 'r')
        Lines = filer.readlines()
        NewLines = Lines
        filer.close()

        filew = open(BoloConfigFile, 'w')        

        NewLine1 = "etendues = "
        NewLine2 = "etendue_errors = "
        for i in range(len(self.etendues)):
            NewLine1 = NewLine1 + str(self.etendues[i]) + "\n"
            NewLine2 = NewLine2 + str(self.etendue_errors[i]) + "\n"
            if i != len(self.etendues) - 1:
                NewLine1 = NewLine1 + "\t"
                NewLine2 = NewLine2 + "\t"

        for i in range(len(Lines)):
            if Lines[i][0:6] == "zeta_o":
                NewLines.insert(i+1, NewLine1)
                NewLines.insert(i+2, NewLine2)
            else:
                pass
        
        filew.writelines(NewLines)

        filew.close()
            

    def reformat_Bolo_Config_File(self, BoloConfigFile):
        '''
        This function converts the bolo config files written by Matt Reinke to 
        INI config files. INI config files have supported reading and writing
        modules in python, so this conversion is worthwhile to leverage those
        tools. Returns a filename of a .ini file which is either the same as the
        one passed in (if it was a .ini), or of the newly converted one.
        '''

        if BoloConfigFile[-3:] == 'ini':
            return BoloConfigFile
        else:
            
            filer = open(BoloConfigFile, 'r')
            Lines = filer.readlines()

            fn = BoloConfigFile[:-3] + 'ini'
            filew = open(fn, 'w')        

            NewLines = ['[Metadata]\n']
            skipLine=0

            for i in range(len(Lines)):
                if skipLine:
                    skipLine=0
                else:
                    cleanThisLine = Lines[i].replace('\xa0', ' ')
                    cleanThisLine = cleanThisLine.replace(';', '\ndone')
                    cleanThisLine = cleanThisLine.split('done',1)[0]
                    if cleanThisLine[0:11] == '###NOTES###':
                        cleanNextLine = Lines[i+1].replace('\xa0', ' ')
                        NewLines.append('notes = ' + cleanNextLine)
                        skipLine=1

                    elif cleanThisLine[0:16] == '###DATA START###':
                        NewLines.append('[Data]\n')

                    elif cleanThisLine[0:14] == '###DATA END###':
                        # do nothing
                        pass
                    else:
                        NewLines.append(cleanThisLine)


            filew.writelines(NewLines)

            filer.close()
            filew.close()

            return fn
            

    def read_Bolo_Config_File(self):
        '''
        Here we assume the config file is already in .ini format. Use configparser
        and read in this file to fields of this object.
        '''
        config = configparser.ConfigParser()
        config.read(self.boloConfigFile)

        
        self.name =config['Metadata']['name']
        self.dtype =config['Metadata']['type'] # diagnostic type
        self.notes = config['Metadata']['notes']
        self.ap_ro = float(config['Data']['ap_ro']) # major radius of aperture (m)
        self.ap_zo = float(config['Data']['ap_zo']) # heigh of aperture (m)
        self.ap_alpha = float(config['Data']['ap_alpha']) # aperture rotation angle alpha (radians)
        self.ap_beta = float(config['Data']['ap_beta']) # aperture rotation angle beta (radians)
        self.ap_gamma = float(config['Data']['ap_gamma']) # aperture rotation angle gamma (radians)
        self.ap_y = float(config['Data']['ap_y'])/1e2 # aperture width (meters)
        self.ap_z = float(config['Data']['ap_z'])/1e2 # aperture width (meters)
        self.x0 = np.array(config['Data']['x0'].split(','), dtype=float)/1e2 # detector coordinate system (center) (meters)
        self.x1 = np.array(config['Data']['x1'].split(','), dtype=float)/1e2 # detector coordinate system (zeta) (meters)
        self.x2 = np.array(config['Data']['x2'].split(','), dtype=float)/1e2 # detector coordinate system (xi) (meters)
        self.det_xi = float(config['Data']['det_xi'])/1e2 # detector width (meters)
        self.det_zeta = float(config['Data']['det_zeta'])/1e2 # detector height (meters)
        self.xi = float(config['Data']['xi'])/1e2 # xi locations of detector centers (meters)
        self.zeta = float(config['Data']['zeta'])/1e2 # zeta locations of detector centers (meters)
        self.xi_ch = float(config['Data']['xi_ch'])/1e2 # xi spacing of channels (meters)
        self.n_xi = int(config['Data']['n_xi']) # number of xi channels
        self.xi_o = float(config['Data']['xi_o'])/1e2 # xi location of center channel
        self.zeta_ch = float(config['Data']['zeta_ch'])/1e2 # zeta spacing of channels (meters)
        self.n_zeta = int(config['Data']['n_zeta']) # number of zeta channels
        self.zeta_o = float(config['Data']['zeta_o'])/1e2 # zeta location of center channel
        
        try:
            etendues = config['Data']['etendues'].splitlines()
            self.etendues = [float(x) for x in etendues]
            etendue_errors = config['Data']['etendues'].splitlines()
            self.etendue_errors = [float(x) for x in etendue_errors]
        except:
            pass
        

class CherabBasicBolometer(Bolometer):
    # Child class of Bolometer. This is the basic bolometer setup given in Cherab tutorials. 

    def __init__(self, BoloConfigFile, World, IndicatorLights):
        super(CherabBasicBolometer,self).__init__(BoloConfigFile=BoloConfigFile, World=World,\
             IndicatorLights=IndicatorLights)
        
        self.build()
        
        # If config file does not already have etendue, calculate etendue and rewrite config file with it
        if self.etendues == None:
            print("Calculating an etendue")
            self.calc_etendues()
            self.add_etendue_configfile(BoloConfigFile = BoloConfigFile)
        
    def build(self):

        # Convenient constants
        # XAXIS = Vector3D(1, 0, 0) #unused
        YAXIS = Vector3D(0, 1, 0)
        ZAXIS = Vector3D(0, 0, 1)
        ORIGIN = Point3D(0, 0, 0)

        xivec = (self.x2 - self.x0)/np.linalg.norm(self.x2 - self.x0)
        zetavec = (self.x1 - self.x0)/np.linalg.norm(self.x1 - self.x0)
        xivec = Vector3D(xivec[0], xivec[1], xivec[2])
        zetavec = Vector3D(zetavec[0], zetavec[1], zetavec[2])
        
        # Bolometer geometry
        BOX_WIDTH = 0.11
        BOX_HEIGHT = 0.11
        BOX_DEPTH = 0.2
        SLIT_WIDTH = self.ap_y
        SLIT_HEIGHT = self.ap_z
        FOIL_WIDTH = self.det_xi
        FOIL_HEIGHT = self.det_zeta
        FOIL_CORNER_CURVATURE = 0.0005

        world = self.world

        ########################################################################
        # Build a bolometer camera from a configuration file
        ########################################################################


        # To position the camera relative to its parent, set the `transform`
        # property to produce the correct translation and rotation.
        camera_box = Box(lower=Point3D(-BOX_DEPTH, -BOX_WIDTH / 2, -BOX_HEIGHT / 2),
                         upper=Point3D(0, BOX_WIDTH / 2, BOX_HEIGHT / 2))
        # Hollow out the box
        outside_box = Box(lower=camera_box.lower - Vector3D(1e-5, 1e-5, 1e-5),
                          upper=camera_box.upper + Vector3D(1e-5, 1e-5, 1e-5))
        camera_box = Subtract(outside_box, camera_box)
        # The slit is a hole in the box
        aperture = Box(lower=Point3D(-1e-4, -SLIT_WIDTH / 2, -SLIT_HEIGHT / 2),
                       upper=Point3D(1e-4, SLIT_WIDTH / 2, SLIT_HEIGHT / 2))
        camera_box = Subtract(camera_box, aperture)
        camera_box.material = AbsorbingSurface() #UniformSurfaceEmitter(red, 100.0)
        # Instance of the bolometer camera
        bolometer_camera = BolometerCamera(camera_geometry=camera_box, parent=world,
                                           name="Demo camera")
        # The bolometer slit in this instance just contains targeting information
        # for the ray tracing, since we have already given our camera a geometry
        # The slit is defined in the local coordinate system of the camera
        slit = BolometerSlit(slit_id="Example slit", centre_point=ORIGIN,
                             basis_x=YAXIS, dx=SLIT_WIDTH, basis_y=ZAXIS, dy=SLIT_HEIGHT,
                             parent=bolometer_camera)
        
        sensor = Node(name="Bolometer sensor", parent=bolometer_camera,
                      transform=translate(self.x0[0], self.x0[1], self.x0[2]))


        # if the number of xi channels is odd, there is a central channel. otherwise, there is no central channel.
        # this one expression should capture both cases.
        xiArray = np.linspace(self.xi_o - self.xi_ch*(self.n_xi-1)/2., self.xi_o + self.xi_ch*(self.n_xi-1)/2., num=self.n_xi)


        for i in range(self.n_xi):
            xi = xiArray[i]
            foil_transform = translate(xivec[0]*xi,xivec[1]*xi, xivec[2]*xi) * sensor.transform

            # pretty sure that the basis settings below are now correct and in agreement
            # with Matt's thesis. might need to do the same to the slit above.
            foil = BolometerFoil(detector_id="Foil {}".format(i + 1),
                                 centre_point=ORIGIN.transform(foil_transform),
                                 basis_x=xivec, dx=FOIL_WIDTH,
                                 basis_y=zetavec, dy=FOIL_HEIGHT,
                                 slit=slit, parent=bolometer_camera, units="Power",
                                 accumulate=False, curvature_radius=FOIL_CORNER_CURVATURE)
            bolometer_camera.add_foil_detector(foil)


        # the last step here is to translate from alpha, beta, gamma rotations to a single Affine matrix.
        # we know that we can get the correct viewing geometry by setting the forward direction and the up
        # direction such that the cross-product of up x forward is in the desired direction. Recall that the
        # rotate_basis function is defined like rotate_basis(forward, up). So, we will start with a vector in
        # the +x direction (our forward) and in the +z direction (our up), rotate it with alpha beta and gamma,
        # and then find the Cherab up x forward that gives the same result.
        ourForward = np.array([1,0,0])
        ourUp = np.array([0,0,1])
        alphaMat = Rot.from_euler('x', self.ap_alpha)
        alphaMat = alphaMat.as_matrix()
        betaMat = Rot.from_euler('y', -self.ap_beta)
        betaMat = betaMat.as_matrix()
        gammaMat = Rot.from_euler('z', self.ap_gamma)
        gammaMat = gammaMat.as_matrix()

        # rotating the forward vector
        ourForwardR = np.matmul(alphaMat,ourForward)
        ourForwardR = np.matmul(betaMat,ourForwardR)
        ourForwardR = np.matmul(gammaMat,ourForwardR)

        # rotating the up vector
        ourUpR = np.matmul(alphaMat, ourUp)
        ourUpR = np.matmul(betaMat, ourUpR)
        ourUpR = np.matmul(gammaMat, ourUpR)

        # ourForwardR is now the direction we would like to look. that means CherabUp x CherabForward = ourForwardR.
        # not enough constraints yet, we also need to use ourUpR. Last constraint is that ourUpR should be
        # identically CherabForward. So we have CherabUp x ourUpR = ourForwardR.
        cherabUp = np.cross(ourUpR, ourForwardR, axis=None)
        ourUpR = Vector3D(ourUpR[0], ourUpR[1], ourUpR[2])
        cherabForward = ourUpR
        cherabUp = Vector3D(cherabUp[0], cherabUp[1], cherabUp[2])

        # we've got the camera properly aligned, but we are not translating properly. we need to rotate the R Z translation
        # to get the proper XYZ.
        transVec = np.array([self.ap_ro, 0., 0.])
        gammaMatTrans = Rot.from_euler('z', self.ap_gamma - np.pi)
        gammaMatTrans = gammaMatTrans.as_matrix()
        transVec = np.matmul(gammaMatTrans, transVec)
        
        bolometer_camera.transform = translate(transVec[0],transVec[1],self.ap_zo)\
            *rotate_basis(cherabForward, cherabUp) 

        
        self.bolometer_camera = bolometer_camera