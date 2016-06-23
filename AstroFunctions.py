## AstroFunctions.py
# ------------------------- #
# Description:
# Various Python functions useful for astrodynamics applications.
# Most of these functions are based on Fundamentals of Astrodynamics, Vallado
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits:
# ------------------------- #
# Import libraries
import numpy as np
import AstroConstants as AstCnst

# -------------------------- #
def fnKepsToCarts(a,e, i, omega,BigOmega, nu):
    # semi-parameter 
    p = a*(1 - e**2);
    # pqw: perifocal coordinate system 
    # Find R and V in pqw coordinate system
    R_pqw = np.zeros([3],dtype=np.float64);
    V_pqw = np.zeros([3],dtype=np.float64);
    
    R_pqw[0] = p*np.cos(nu)/(1 + e*np.cos(nu));
    R_pqw[1] = p*np.sin(nu)/(1 + e*np.cos(nu));
    R_pqw[2] = 0;  
    V_pqw[0] = -np.sqrt(AstCnst.mu_E/p)*np.sin(nu);
    V_pqw[1] =  np.sqrt(AstCnst.mu_E/p)*(e + np.cos(nu));
    V_pqw[2] =   0;
    # Then rotate into ECI frame
    R = np.dot(np.dot(fnRotate3(-BigOmega),fnRotate1(-i)),np.dot(fnRotate3(-omega),R_pqw));
    V = np.dot(np.dot(fnRotate3(-BigOmega),fnRotate1(-i)),np.dot(fnRotate3(-omega),V_pqw));
    xstate = np.hstack((R,V));
    return xstate # Validated against Vallado's example 2-3. 20/06/16
# -- Rotation functions  --------------------------------#
def fnRotate1(alpha_rad):
    T = np.array([[1,                 0   ,          0],
                  [0,  np.cos(alpha_rad)  ,np.sin(alpha_rad)],
                  [0, -np.sin(alpha_rad)  ,np.cos(alpha_rad)]],dtype=np.float64);
    return T # Validated against Vallado's example 2-3. 20/06/16

def fnRotate2(alpha_rad):
    T = np.array([[np.cos(alpha_rad), 0, -np.sin(alpha_rad)], 
                  [                0, 1,                  0],
                  [np.sin(alpha_rad), 0,  np.cos(alpha_rad)]],dtype=np.float64);
    return T

def fnRotate3(alpha_rad):
    T = np.array([[ np.cos(alpha_rad),np.sin(alpha_rad),0], 
                  [-np.sin(alpha_rad),np.cos(alpha_rad),0],
                  [                 0,                0,1]],dtype=np.float64);
    return T # Validated against Vallado's example 2-3. 20/06/16
# --Frame conversion functions------------------------------------------------ #
##def fnECItoECEF():
##ECEF = zeros(3,size(ECI,2));
##
##if nargin == 3 % if # of arguments is 3
##    V_ECEF = zeros(size(ECI,1),size(ECI,2));
##end
##
##% Loop through all observations
##for index = 1:size(ECI,2) 
##    % Rotating the ECI vector into the ECEF frame via the GST angle about the Z-axis
##    ECEF(:,index) = fnRotate3(theta_GST(index))*ECI(:,index);
##    if nargin == 3
##        V_ECEF(:,index) = fnRotate3(theta_GST(index))*V_ECI(:,index);
##    end
##end
