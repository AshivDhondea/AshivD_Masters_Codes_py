## ObservationModels.py
# ------------------------- #
# Description:
# A collection of functions which implement observation functions.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 23 June 2016
# Edits:
#
# ------------------------- #
# Theoretical background:
# 1. Tracking Filter Engineering, Norman Morrison. 2013
# 2. Fundamentals of astrodynamics. Vallado. 2001
# ------------------------- #
# Import libraries
import numpy as np
from numpy import linalg
import scipy.linalg
# ---------------------------------------- #
## Observation Model functions

def fnH(Xinput):
    # Nonlinear measurement function.
    # Sensor measures range and look angles.
    Xinput_polar = np.zeros([3],dtype=np.float64);
    # Calculate range
    Xinput_polar[0] = np.linalg.norm(Xinput);
    # Calculate elevation
    Xinput_polar[1] = np.arctan(Xinput[2]/np.linalg.norm(Xinput[0:2]));
    # Calculate azimuth
    Xinput_polar[2] = np.arctan2(Xinput[0],Xinput[1]);
    return Xinput_polar

def fnHinv(Xinput):
    # Xinput[0] = range
    # Xinput[1] = elevation
    # Xinput[2] = azimuth
    Xout = np.zeros([3],dtype=np.float64);
    Xout[0] = Xinput[0]*np.cos(Xinput[1])*np.sin(Xinput[2]); # x
    Xout[1] = Xinput[0]*np.cos(Xinput[1])*np.cos(Xinput[2]); # y
    Xout[2] = Xinput[0]*np.sin(Xinput[1]); # z
    return Xout

def fnJacobianH(Xnom):
    # Jacobian of nonlinear measurement function fnH
    # Xnom[0] = range
    # Xnom[1] = elevation
    # Xnom[2] = azimuth
    rho = np.linalg.norm(Xnom);
    s = np.linalg.norm(Xnom[0:2]);
    Mmatrix = np.zeros([3,3],dtype=np.float64);
    Mmatrix[0,0] = Xnom[0]/rho; 
    Mmatrix[0,1] = Xnom[1]/rho; 
    Mmatrix[0,2] = Xnom[2]/rho; 

    Mmatrix[1,0] = -Xnom[0]*Xnom[2]/(s*rho**2);
    Mmatrix[1,1] = -Xnom[1]*Xnom[2]/(s*rho**2); 
    Mmatrix[1,2] = s/(rho**2);

    Mmatrix[2,0] = -Xnom[1]/s**2;
    Mmatrix[2,1] = Xnom[0]/s**2;           
    return Mmatrix
