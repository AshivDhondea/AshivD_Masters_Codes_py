## Pj000_Orbit_Propagation.py
# ------------------------- #
# Description:
# This project investigates orbit propagation.
# We first convert the Keplerians of the ISS to a Cartesian state vector.
# We then propagate the state vector over time through a J2-perturbed Keplerian
# model. The numerical integration scheme employed is the classical
# 4th order Runge-Kutta method.
# Thereafter, we develop the corresponding function for the state transition matrix.
# We also perform numerical integration of this nonlinear function.
# To this end, we augment the state vector by the elements of the STM reshaped
# into a vector.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 23 June 2016
# Edits:
# ------------------------- #
# Theoretical background
# 1. Tracking Filter Engineering, Norman Morrison. 2013.
# 2. Statistical Orbit Determination. Tapley, Schutz, Born. 2004
# 3. Fundamentals of astrodynamics and applications. Vallado. 2001
# ------------------------- #
# Import libraries
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# Import Ashiv's own stuff
import AstroConstants as AstCnst
import AstroFunctions as AstFn
import DynamicsFunctions as DynFn
import Num_Integ as ni
# --------------------------- #
# Keplerians extracted from TLE file of ISS on 20 June 2016.
# Keplerians were extracted from the TLE file in MATLAB.
a = 6781.36800424;
e = 0.00005370;
i = np.deg2rad(51.64540000);
BigOmega = np.deg2rad(43.38260000);
omega = np.deg2rad(0.08780000);
nu = np.deg2rad(5.88983219);
# Convert Keplerians to Cartesian state vector
xstate = AstFn.fnKepsToCarts(a,e, i, omega,BigOmega,nu);
# ------------------------------ #
# Declare simulation parameters
dt = 10 # stepsize in [s]
t = np.arange(0,60*60*2,dt,dtype=float) # simulation period
# Note that the ISS has a period of roughly 90 minutes.
# We are simulating here for a period of 2[h].

# Ystate is the state vector over the entire time interval
Ystate = np.zeros((6,len(t)),dtype = np.float64);
Ystate[:,0] = xstate;

for index in range(1,len(t)):
    # Perform RK4 numerical integration on the J2-perturbed dynamics.
    Ystate[:,index] = ni.fnRK4_vector(DynFn.fnKepler_J2,dt,Ystate[:,index-1],t[index-1]);
# ------------------------------------------------- #
# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")
mpl.rcParams['legend.fontsize'] = 10

ax = fig.gca(projection='3d')
ax.plot(Ystate[0,:], Ystate[1,:], Ystate[2,:], label='ISS orbit')
ax.legend()

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')

plt.show()
# ------------------------------------------------- #
