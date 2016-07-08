## Pj007_Simulating_TwoBody_SDE.py
# Testing SRK integration on the J2-perturbed two body problem.
# ------------------------- #
# Description:
# This project demonstrates numerical integration of the two-body problem plus process noise.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 06 July 2016
# Edits: 
# ------------------------- #
# Theoretical background
# 1. Bayesian Signal Processing, Sarkka. 2013.
# 2. ekfukf toolbox documentation
# 3. sdeint package in Python: https://pypi.python.org/pypi/sdeint
# ------------------------- #
# Import libraries
import numpy as np
import sdeint
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
# Import Ashiv's own stuff
import AstroConstants as AstCnst
import DynamicsFunctions as DynFn
# ------------------------------------------------- #
## Declare variables
# Dispersion matrix
L = np.zeros([6,3],dtype=np.float64);
L[3,0] = 1.0; L[4,1]=1.0;L[5,2]=1.0;

t = 0;
true_m0 = np.array([4600.53917168,4950.79940098,553.772365323,-3.83039035655,2.89058678525,5.9797711844 ],dtype=np.float64);
true_P0 = np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]);

x = true_m0 + np.random.multivariate_normal([0,0,0,0,0,0],true_P0);

dt = 10;
nsteps = 60*60*2;

true_Qc = np.diag([2.4064e-4,2.4064e-4,2.4064e-4]) ;

B = dt*true_Qc;

def f(x,t):
    return DynFn.fnKepler_J2(t,x);

def G(x, t):
    return np.dot(L,B)

tspan = np.arange(0,nsteps,dt);
result = sdeint.itoint(f, G, x, tspan);
result2 = sdeint.itoSRI2(f,G,x,tspan);
print 'done'
# ------------------------------------------------------------------------ #
# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")
mpl.rcParams['legend.fontsize'] = 10

ax = fig.gca(projection='3d')
ax.set_axis_bgcolor('lightsteelblue')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = AstCnst.R_E* np.outer(np.cos(u), np.sin(v))
y = AstCnst.R_E* np.outer(np.sin(u), np.sin(v))
z = AstCnst.R_E* np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='mediumseagreen')

ax.plot(result[:,0], result[:,1], result[:,2],color='maroon',linewidth='2',label='ISS orbit')
ax.legend()

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')

plt.show()
##
fig = plt.figure()
fig.suptitle('Euler-Maruyama & Stochastic Runge-Kutta')
plt.plot(tspan,result[:,1],'b.',label='em y');
plt.plot(tspan,result[:,0],'r.',label='em x');
plt.plot(tspan,result[:,2],'y.',label='em z');
plt.plot(tspan,result2[:,0],'g.',label='srk x');
plt.plot(tspan,result2[:,1],'m.',label='srk y');
plt.plot(tspan,result2[:,2],'k.',label='srk z');
plt.xlabel('t [s]')
plt.ylabel('displacement [km]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

