## AstroFunctions.py
# ------------------------- #
# Description:
# Various Python functions useful for astrodynamics applications.
# Most of these functions are based on Fundamentals of Astrodynamics, Vallado. 4th ed.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits: 02 August 2016: added a number of functions for coordinate transformations, etc from Vallado book.
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

def fnKeplerOrbitalPeriod(a):
    # T is the orbital period in [s]
    T = 2*np.pi*np.sqrt(a**3/AstCnst.mu_E);
    return T

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
## 3 August 2016: note that these are only approximate transforms. Precession,
# nutation, polar motion are ignored. Can be used in simulations, but not for real-life applications.
def fnECItoECEF(ECI,theta):
    ECEF = np.zeros([3],dtype=np.float64);
    # Rotating the ECI vector into the ECEF frame via the GST angle about the Z-axis
    ECEF = np.dot(fnRotate3(theta),ECI);
    return ECEF
def fnECEFtoECI(ECEF,theta):
    ECI = np.zeros([3],dtype=np.float64);
    # Rotating the ECEF vector into the ECI frame via the GST angle about the Z-axis
    ECI = np.dot(fnRotate3(-theta),ECEF);
    return ECI

# -- Algorithms from Vallado, 4th edition ---------------------------------------------- #
## Refer to algorithm 51 in Vallado, 4th ed.
def fnSiteTrack(latitude_gd,longitude,altitude,rho,az,el):
    """
    fnSiteTrack implements Algorithm 51 in Vallado, 4th edition.
    """
    # Find radar sensor's location in the ECEF frame.
    Radar_ECEF = fnRadarSite(latitude_gd,longitude,altitude);
    # Convert observations in the radar centered (local) coordinate frame to the SEZ Cartesian frame.
    R_SEZ = fnRadarToSEZ(rho,az,el );
    # Convert position in SEZ frame to ECEF frame.
    R_ECEF = fnSEZtoECEF( R_SEZ,latitude_gd,longitude );

    r_ecef = Radar_ECEF + R_ECEF;
    # Convert r_ecef to R_ECI using algorithm 24.

    return R_ECI

##def fnFK5():
##    """
##    fnFK5 implements Algorithm 24 in Vallado and performs FK5 reduction.
##    02 August 2016. Called by fnSiteTrack.
##    """
##    return R_ECI

def fnRadarSite(latitude_gd,longitude,altitude):
    """
    % fnRadarSite determines the ECEF position of the radar tracking station.
    % It emulates site.m from Vallado's book. [part of Algorithm 51].
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    # Earth's shape eccentricity
    e_earth_squared = (2*AstCnst.flattening-AstCnst.flattening**2);

    # Calculate 2 auxiliary points
    cearth= AstCnst.R_E/np.sqrt( 1 - (e_earth_squared*(np.sin( latitude_gd ))**2 ) );
    searth = cearth*(1-e_earth_squared);

    h_ellp = fnEllipsoidalHeight( altitude,latitude_gd ); #  Validated 02/08/16 with example 7-1 in Vallado book.
    #%h_ellp = altitude;
    rdel  = (cearth+ h_ellp )*np.cos( latitude_gd);
    rk    = (searth + h_ellp )*np.sin( latitude_gd );

    # Radar_ECEF is the position vector in the Cartesian ECEF frame.    
    Radar_ECEF = np.zeros(3,dtype=np.float64);
    Radar_ECEF[0] = rdel * np.cos( longitude );
    Radar_ECEF[1] = rdel * np.sin( longitude );
    Radar_ECEF[2] = rk;
    return Radar_ECEF

def fnEllipsoidalHeight(altitude,latitude_gd):
    """
    % fnEllipsoidalHeight finds the ellipsoidal height for a target at a given
    % altitude
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    latitude_gc = fnGeodeticToGeocentricLatitude(latitude_gd);
    h_ellp = altitude/(np.cos(latitude_gd - latitude_gc));
    return h_ellp

def fnGeodeticToGeocentricLatitude(latitude_gd):
    """
    % fnGeodeticToGeocentricLatitude converts the latitude from geodetic to
    % geocrentric. This is necessary because of the oblateness of the Earth.
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    latitude_gc = np.arctan((1-AstCnst.flattening**2)*np.tan(latitude_gd));
    return latitude_gc

def fnRadarToSEZ(rho,az,el ):
    """
    % fnRadarToSEZ converts observations (rho,az,el) to the SEZ Cartesian
    % coordinate system.
    % Based on raz2rvs.m from Vallado's book

    % Note that 
    % 1. angles are assumed to be in degrees in this function. [02/08/16: no, in radians.]
    % 2. the azimuth angle is measured from the negative x axis to the positive
    % y axis in the xy plane.
    % 3. the elevation angle is measured from the positive xy plane to the
    % positive z axis.
    # Validated with example 7-1
    """
    R_SEZ = np.zeros(3,dtype=np.float64);
    R_SEZ[0] = -rho*np.cos(el)*np.cos(az); #% x
    R_SEZ[1] =  rho*np.cos(el)*np.sin(az); #% y
    R_SEZ[2] =  rho*np.sin(el);      #  % z
    return R_SEZ

def fnSEZtoECEF( R_SEZ,latitude_gd,longitude ):
    """
    %% fnSEZtoECEF transforms a position vector in the SEZ frame whose origin
    % is at a tracking station with coordinates (latitude,longitude) on Earth
    % to the ECEF frame.
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    R_ECEF = np.dot(np.dot(fnRotate3(-longitude),fnRotate2(-(0.5*np.pi - latitude_gd))),R_SEZ);
    return R_ECEF
