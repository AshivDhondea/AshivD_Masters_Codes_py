## Pj015_CD_EKF_TwoBody_SDE.py
# 
# ------------------------- #
# Description:
# This project shows CD-EKF methods filtering radar data generated in Pj014_Simulating_TwoBody_save_data_radar.py
# Various EKFs are employed to process the radar data.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 27 July 2016
# Edits: 28 July 2016: edited function calls for the CD-EKF predict functions.
# ------------------------- #
# Theoretical background
# 1. Tracking Filter Engineering, Norman Morrison. 2013 aka TFE
# 2. Statistical Orbit Determination, Tapley, Schutz, Born. 2004 aka SOD
# 3. Bayesian Signal Processing. Sarkka. 2013 and ekfukf toolbox.
# 4. Various ways to compute the continuous-discrete extended Kalman filter. Frogerais, 2012. IEEE Transactions on Automatic Control.
# 5. Cubature Kalman filters for continuous-time dynamic models Part II: A solution based on moment matching. Crouse,2014.

""" Based on "Various ways to compute the continuous-discrete extended Kalman filter"
     @article{frogerais2012various,
      title={Various ways to compute the continuous-discrete extended Kalman filter},
      author={Frogerais, Paul and Bellanger, Jean-Jacques and Senhadji, Lotfi},
      journal={Automatic Control, IEEE Transactions on},
      volume={57},
      number={4},
      pages={1000--1004},
      year={2012},
      publisher={IEEE}
    } """
""" Based on "Cubature Kalman filters for continuous-time dynamic models Part II: A solution based on moment matching"
    @INPROCEEDINGS{6875583, 
    author={D. F. Crouse}, 
    booktitle={2014 IEEE Radar Conference}, 
    title={Cubature Kalman filters for continuous-time dynamic models Part II: A solution based on moment matching}, 
    year={2014}, 
    pages={0194-0199}, 
    month={May}
    }"""
# ------------------------- #
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
# Import Ashiv's own stuff
import AstroConstants as AstCnst
import AstroFunctions as AstFn
import DynamicsFunctions as DynFn
import Num_Integ as ni
import ObservationModels as Obs
import Filters as Flt
import MathsFunctions as Mth
# --------------------------- #
## Read true data from files
dt = 0.0001;#[s] 29.7.16
t_end = 4;#[s] Target observed only for 2 seconds. 29.7.16
del_y =100;#decimation in time.i.e. measurements are sampled at a rate 10 times smaller than x.
t_y = np.arange(0,t_end,dt*del_y,dtype=np.float64); # time vector for measurements.
# Second, extract true data
Xradar = np.zeros([3,len(t_y)],dtype=np.float64);
index = 0;
with open('Pj014_Simulating_TwoBody_save_data_radar.txt') as fp:
    for line in fp:
        data = line.split();
        # Place the data read in from the file into the array of state vectors.
        Xradar[:,index] = [np.float64(i) for i in data];
        index+=1;
    
fp.close();
print 'Radar data read in.'
# ------------------------------------------------- #
## Variables required by the filters
# Process noise covariance matrix. continuous-time and discretized version.
Qc = np.diag([2.4064e-4,2.4064e-4,2.4064e-4]) ; # can be tuned as desired.
Qd = dt*Qc;
# Measurement noise covariance matrix
R = np.diag([np.square(50e-3),np.square(np.deg2rad(0.1)),np.square(np.deg2rad(0.1))]); # R contains the variance in the range,azimuth and elevation readings.
# Initial state estimate and initial covariance matrix estimate
true_m0 = np.array([2085,-3686,5293,6.3,4.3,0.5],dtype=np.float64);
true_P0 = np.diag([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]);
# Dispersion matrix.
L = np.zeros([6,3],dtype=np.float64);
L[3,0] = 1.0; L[4,1]=1.0;L[5,2]=1.0;

# Initialize m and P.
m = true_m0 + np.random.multivariate_normal(np.zeros(6,dtype=np.float64),true_P0);
P = true_P0;
# arrays to hold estimates.
x_hat = np.zeros([6,len(t_y)],dtype=np.float64);
P_hat = np.zeros([6,6,len(t_y)],dtype=np.float64);
# --------------------------------------------------------- #
## Filtering
x_hat,P_hat = Flt.fnEKF(m,P,Xradar,R,dt,del_y,t_y,100,Qc,L,'1');
x_hat_n,P_hat_n = Flt.fnEKF(m,P,Xradar,R,dt,del_y,t_y,1,Qc,L,'2');
x_hat_euler,P_hat_euler = Flt.fnEKF(m,P,Xradar,R,dt,del_y,t_y,1,Qc,L,'3');
x_hat_srk2,P_hat_srk2 = Flt.fnEKF(m,P,Xradar,R,dt,del_y,t_y,1,Qc,L,'4');
x_hat_srk4,P_hat_srk4 = Flt.fnEKF(m,P,Xradar,R,dt,del_y,t_y,1,Qc,L,'5');
print 'Filtering done.'
# ------------------------------------------------------------- #
# Read true data
xtrue = np.zeros([6,len(t_y)],dtype=np.float64); # array of state vectors of the time interval of interest.
index = 0;
with open('Pj014_Simulating_TwoBody_save_data_radar_true.txt') as fp:
    for line in fp:
        data = line.split();
        # Place the data read in from the file into the array of state vectors.
        xtrue[:,index] = [np.float64(i) for i in data];
        index+=1;
            
fp.close();
# ------------------------------------------------------------- #
#### Evaluate results
##fig = plt.figure()
##fig.suptitle('True vs estimated x-position')
##plt.plot(t_y [0:len(t_y)],xtrue[0,:],'b.',label='x true');
##plt.plot(t_y [0:len(t_y)],x_hat[0,:],'b*',label='x estimated');
##plt.xlabel('t [s]')
##plt.ylabel('position [km]')
##plt.legend(loc='upper left')
##ax = plt.gca()
##ax.grid(True)
##plt.show()
##
##fig = plt.figure()
##fig.suptitle('True vs estimated y-position')
##plt.plot(t_y [0:len(t_y)],xtrue[1,:],'r.',label='y true');
##plt.plot(t_y [0:len(t_y)],x_hat[1,:],'r*',label='y est');
##plt.xlabel('t [s]')
##plt.ylabel('position [km]')
##plt.legend(loc='upper left')
##ax = plt.gca()
##ax.grid(True)
##plt.show()
##
##fig = plt.figure()
##fig.suptitle('True vs estimated z-position')
##plt.plot(t_y [0:len(t_y)],xtrue[2,:],'y.',label='z true');
##plt.plot(t_y [0:len(t_y)],x_hat[2,:],'y*',label='z est');
##plt.xlabel('t [s]')
##plt.ylabel('position [km]')
##plt.legend(loc='upper left')
##ax = plt.gca()
##ax.grid(True)
##plt.show()
##
##fig = plt.figure()
##fig.suptitle('True vs estimated x-velocity')
##plt.plot(t_y [0:len(t_y)],xtrue[3,:],'b.',label='xdot true');
##plt.plot(t_y [0:len(t_y)],x_hat[3,:],'b*',label='xdot est');
##plt.xlabel('t [s]')
##plt.ylabel('velocity [km/s]')
##plt.legend(loc='upper left')
##ax = plt.gca()
##ax.grid(True)
##plt.show()
##
##fig = plt.figure()
##fig.suptitle('True vs estimated y-velocity')
##plt.plot(t_y [0:len(t_y)],xtrue[4,:],'b.',label='ydot true');
##plt.plot(t_y [0:len(t_y)],x_hat[4,:],'b*',label='ydot est');
##plt.xlabel('t [s]')
##plt.ylabel('velocity [km/s]')
##plt.legend(loc='upper left')
##ax = plt.gca()
##ax.grid(True)
##plt.show()
##
##fig = plt.figure()
##fig.suptitle('True vs estimated z-velocity')
##plt.plot(t_y [0:len(t_y)],xtrue[5,:],'b.',label='zdot true');
##plt.plot(t_y [0:len(t_y)],x_hat[5,:],'b*',label='zdot est');
##plt.xlabel('t [s]')
##plt.ylabel('velocity [km/s]')
##plt.legend(loc='upper left')
##ax = plt.gca()
##ax.grid(True)
##plt.show()

# ----------------------------------------------------------------- #
## Calculate RMSE
rmse_pos = np.zeros(len(t_y),dtype=np.float64);
rmse_vel = np.zeros(len(t_y),dtype=np.float64);
diff = 0;
diff_n = 0;
diff_euler = 0;
diff_srk2 = 0;
diff_srk4 = 0;

print 'Calculating RMSE';
#print np.dot(np.diag(P_hat[:,:,0]),np.eye(np.shape(x_hat)[0]))
                                          
for index in range(0,len(t_y)):
    diff += np.sum(np.square(x_hat[:,index] - xtrue[:,index]));
    diff_n += np.sum(np.square(x_hat_n[:,index] - xtrue[:,index]));
    diff_euler += np.sum(np.square(x_hat_euler[:,index] - xtrue[:,index]));
    diff_srk2 += np.sum(np.square(x_hat_srk2[:,index] - xtrue[:,index]));
    diff_srk4 += np.sum(np.square(x_hat_srk4[:,index] - xtrue[:,index]));
##    rmse_pos[index] = np.linalg.norm(x_hat[0:3,index] - xtrue[0:3,index]) / np.sqrt(len(t_y));
##    rmse_vel[index] = np.linalg.norm(x_hat[3:,index] - xtrue[3:,index]) / np.sqrt(len(t_y));

rmse = np.sqrt(diff)/np.sqrt(len(t_y)*3.0);
print rmse
rmse_n = np.sqrt(diff_n)/np.sqrt(len(t_y)*3.0);
print rmse_n
rmse_euler = np.sqrt(diff_euler)/np.sqrt(len(t_y)*3.0);
print rmse_euler
rmse_srk2 = np.sqrt(diff_srk2)/np.sqrt(len(t_y)*3.0);
print rmse_srk2
rmse_srk4 = np.sqrt(diff_srk4)/np.sqrt(len(t_y)*3.0);
print rmse_srk4    
####    
##fig = plt.figure()
##fig.suptitle('Plot of RMSE in position')
##plt.plot(t_y[0:len(t_y)],rmse_pos[:],'k',);
##plt.xlabel('t [s]');
##plt.ylabel('RMSE [km]');
##ax = plt.gca()
##ax.grid(True)
##plt.show()
##
##fig = plt.figure()
##fig.suptitle('Plot of RMSE in velocity')
##plt.plot(t_y[0:len(t_y)],rmse_vel[:],'k',);
##plt.xlabel('t [s]');
##plt.ylabel('RMSE [km/s]');
##ax = plt.gca()
##ax.grid(True)
##plt.show()
