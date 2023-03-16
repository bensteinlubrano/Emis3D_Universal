# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:33:45 2021

@author: bemst
"""

import math
import numpy as np

""" For Chi^2 to p value conversion """
from scipy.stats import chi2

def XY_To_RPhi(X,Y, TorOffset=0.0):
    
    ''' Convert from the Cartesian x,y coordinates to the major radius and toroidal angle phi. 
    Phi is returned in radians and R in meters'''
    
    R = math.hypot(X, Y)
    phi = math.atan2(Y, X) - TorOffset

    if phi < -np.pi:
        phi = phi + 2.*np.pi
    elif phi > np.pi:
        phi = phi - 2.*np.pi
    
    return R, phi

def RPhi_To_XY(R, Phi):
    
    x = R * math.cos(Phi)
    y = R * math.sin(Phi)
    
    return x, y

def RedChi2_To_Pvalue(RedChi2, Dof):
    return chi2.cdf(x=RedChi2*Dof, df=Dof)

def Pvalue_To_RedChi2(Pvalue, Dof):
    return chi2.ppf(q=Pvalue, df=Dof) / Dof