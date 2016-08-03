## Filters.py
# ------------------------- #
# Description:
# A collection of functions which implement functions
# required by different filtering strategies.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 23 June 2016
# Edits:
# 7 July 2016: added functions for Gauss-Newton in the batch mode.
# 8 July 2016: removed function calls from parameters, added function for Gauss-Newton in fixed-memory length mode.
# 10 July 2016: added functions for the EKF operating on nonlinear dynamic systems.
# 16 July 2016: added functions to implement CD-EKF, specifically the 'MC-RK4' algorithm.
# 18 July 2016: added functions to implement the CD-EKF in the 'D-Euler','D-SRK2' and 'D-SRK4' variants
# 21 July 2016: modified fnUT_sigmas. Numpy's Cholesky decomposition only deals with positive definite matrices.
#               However, the covariance matrix may become positive semidefinite. This is why schol from ekfukf toolbox was implemented to decompose PSD matrices.
# 22 July 2016: created the function fnCD_UKF_predict_M for CD-UKF predict step: integrating the sigma points through the dynamics function by RK4
# 24 July 2016: changed all numpy arrays of type float to np.float64.
# 29 July 2016: cleaned up code for the fnCD_EKF_predict_STM_RK4, included function call to generate STM in DynFn.
# 31 July 2016: created the function fnCD_EKF which can call the various versions of the CD-EKF.
# 31 July 2016: cleaned up all the CD-EKF functions.
# ------------------------- #
# Theoretical background:
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
from numpy import linalg
# Include Ashiv's own codes
import DynamicsFunctions as DynFn
import Num_Integ as ni
import ObservationModels as Obs
import MathsFunctions as MathsFn

# ---------------------------------------- #
## Continuous-Discrete Filtering with the Extended Kalman Filter
# MC-RK4 algorithm from Frogerais 2012
def fnCD_EKF_predict_MC_RK4( m,P,dt,steps,Qc,L):
    # fnCD_EKF_predict_MC_RK4 implements the continuous-discrete extended Kalman Filter predict step.
    # This is the MC-RK4 method: Mean-Covariance Runge-Kutta 4th order
    # m is the mean, P is the covariance matrix.
    # Qc is the process noise matrix, L is the dispersion matrix.
    # steps = Number of steps in the RK4 integration
    # dt = time interval from k to k+1.

    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);
    # moment_vector contains the mean vector and covariance matrix which need to be integrated.
    moment_vector = np.zeros((6 + 6*6,len(td)),dtype = np.float64);
    moment_vector[0:6,0] = m; # mean vector
    moment_vector[6:,0] = np.reshape(P,6*6); # covariance matrix 

    for index in range(1,len(td)):
        # Perform RK4 numerical integration on the system of differential equations for the moment.
        moment_vector[:,index] = ni.fnRK4_vector(DynFn.fnMoment_DE,dt/steps,moment_vector[:,index-1],td[index-1],Qc,L);
        
    # Extract the last mean vector and covariance matrix. These correspond to the mean vector and covariance matrix at time index k+1.
    m_pred = moment_vector[0:6,len(td)-1];
    P_pred = np.reshape(moment_vector[6:,len(td)-1],(6,6));
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

# MF-RK4 algorithm from Frogerais 2012
def fnCD_EKF_predict_STM_RK4( m,P,dt,steps,Qc,L):
    # fnCD_EKF_predict_STM_RK4 implements the continuous-discrete extended Kalman Filter predict step.
    # This is the STM-RK4 method: State Transition Matrix propagation by Runge-Kutta 4th order method.
    # m is the mean, P is the covariance matrix.
    # Qc is the process noise matrix, L is the dispersion matrix.
    # steps = Number of steps in the RK4 integration
    # dt = time interval from k to k+1.

    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);
    # augmented state vector contains the state vector and stm which need to be integrated.
    x_aug = np.zeros((6 + 6*6,len(td)),dtype = np.float64);

    x_aug = DynFn.fnGenerate_Nominal_Trajectory(m,td);    
    # Extract the last mean vector and covariance matrix. These correspond to the mean vector and covariance matrix at time index k+1.
    m_pred = x_aug[0:6,len(td)-1];
    stm_end = np.reshape(x_aug[6:,len(td)-1],(6,6));
    #Gamma = np.dot(stm_end,L); # Refer to SOD pg 228. We can do this because L = constant.
    P_pred = np.dot(stm_end,np.dot(P,np.transpose(stm_end))) ;#+ np.dot(Gamma,np.dot(Qc,np.transpose(Gamma)));
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

# D-Euler algorithm from Frogerais 2012
def fnCD_EKF_predict_D_Euler( m,P,dt,steps,Qc,L):
    # fnCD_EKF_predict_D_Euler implements the extended Kalman Filter predict step for Continuous-Discrete Filtering by the D-Euler method.
    # F is the nonlinear dynamics function.
    # A is the Jacobian of the function F evaluated at m.
    # m is the mean, P is the covariance matrix.
    # process noise: Q matrix
    # Edited: 31/07/16: added a for loop so that multiple steps are taken from k to k+1
    
    m_pred = m;
    P_pred = P;
    Qd =  Qc*(dt/steps);
    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);
    
    for index in range (0,steps):
        m_pred = m_pred + (dt/steps)*DynFn.fnKepler_J2(td[index],m_pred);
        A = DynFn.fnJacobian_Kepler_J2(m_pred);
        # D-Euler method approximates the STM by (I + A*dt)
        pseudo_stm = np.eye(6,dtype=np.float64) + (dt/steps)*A;
        P_pred = np.add(np.dot(np.dot(pseudo_stm,P_pred),np.transpose(pseudo_stm)),np.dot(np.dot(L,Qd),np.transpose(L)));

    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

# D-SRK2 algorithm from Frogerais 2012
def fnCD_EKF_predict_D_SRK2( m,P,dt,steps,Qc,L):
    # fnCD_EKF_predict_D_SRK2 implements the extended Kalman Filter predict step for Continuous-Discrete Filtering by the D-SRK2 method.
    # F is the nonlinear dynamics function.
    # A is the Jacobian of the function F evaluated at m.
    # m is the mean, P is the covariance matrix.
    # process noise: Q matrix
    # Edited: 31/07/16: added a for loop so that multiple steps are taken from k to k+1

    m_pred = m;
    P_pred = P;
    Qd =  Qc*(dt/steps);
    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);
    
    for index in range (0,steps):
        # Heun integration
        k1 = DynFn.fnKepler_J2(td[index],m_pred); # fnKepler_J2 does not explicitly contain dt, so it is not critical to have the correct expression for time
        k2 = DynFn.fnKepler_J2(td[index],m_pred + (dt/steps)*k1);
        J1x = DynFn.fnJacobian_Kepler_J2(m_pred);
        J2x = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+(dt/steps)*k1),(np.eye(6,dtype=np.float64) + (dt/steps)*J1x));
        J1w = L/(dt/steps);
        J2w = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+(dt/steps)*k1),(dt/steps)*J1w);

        # Heun. See Frogerais 2012
        m_pred = m_pred + ((dt/steps)/2)*(k1 + k2);
    
        # D-SRK2 method approximates the STM by (I + 0.5*(J1x+J2x)*dt)
        pseudo_stm = np.eye(6,dtype=np.float64) + 0.5*(dt/steps)*(J1x+J2x);
        JacL = 0.5*(dt/steps)*(J1w + J2w);
        P_pred = np.add(np.dot(np.dot(pseudo_stm,P_pred),np.transpose(pseudo_stm)),np.dot(np.dot(JacL,Qd),np.transpose(JacL)));
        # m_pred and P_pred are the predicted mean state vector and covariance
        # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

# D-SRK4 algorithm from Frogerais 2012
def fnCD_EKF_predict_D_SRK4( m,P,dt,steps,Qc,L):
    # fnCD_EKF_predict_D_SRK4 implements the extended Kalman Filter predict step for Continuous-Discrete Filtering by the D-SRK2 method.
    # F is the nonlinear dynamics function.
    # A is the Jacobian of the function F evaluated at m.
    # m is the mean, P is the covariance matrix.
    # process noise: Q matrix
    
    m_pred = m;
    P_pred = P;
    Qd =  Qc*(dt/steps);
    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);

    for index in range (0,steps):
        # Stochastic Runge-Kutta 4 integration
        k1 = DynFn.fnKepler_J2(td[index],m_pred); # fnKepler_J2 does not explicitly contain dt, so it is not critical to have the correct expression for time
        k2 = DynFn.fnKepler_J2(td[index],m_pred + 0.5*(dt/steps)*k1);
        k3 = DynFn.fnKepler_J2(td[index],m_pred + 0.5*(dt/steps)*k2);
        k4 = DynFn.fnKepler_J2(td[index],m_pred +     (dt/steps)*k3);
    
        J1x = DynFn.fnJacobian_Kepler_J2(m_pred);
        J2x = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+0.5*(dt/steps)*k1),(np.eye(6,dtype=np.float64) + 0.5*(dt/steps)*J1x));
        J3x = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+0.5*(dt/steps)*k2),(np.eye(6,dtype=np.float64) + 0.5*(dt/steps)*J2x));
        J4x = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+    (dt/steps)*k3),(np.eye(6,dtype=np.float64) +     (dt/steps)*J3x));

        J1w = L/(dt/steps);
        J2w = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+0.5*(dt/steps)*k1),0.5*(dt/steps)*J1w);
        J3w = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+0.5*(dt/steps)*k2),0.5*(dt/steps)*J2w);
        J4w = np.dot(DynFn.fnJacobian_Kepler_J2(m_pred+    (dt/steps)*k3),    (dt/steps)*J3w);

        # Runge-Kutta 4th order method. See Frogerais 2012
        m_pred = m_pred + ((dt/steps)/6)*(k1 + 2*k2 + 2*k3 + k4);
    
        # D-SRK4 method approximates the STM by this
        pseudo_stm = np.eye(6,dtype=np.float64) + (1/2.0)*(dt/steps)*(J1x+2*J2x+2*J3x+J4x);
        JacL = (1/2.0)*(dt/steps)*(J1w + 2*J2w + 2*J3w + J4w);
        P_pred = np.add(np.dot(np.dot(pseudo_stm,P_pred),np.transpose(pseudo_stm)),np.dot(np.dot(JacL,Qd),np.transpose(JacL)));

    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

########################################################################################################
def fnEKF_predict_kinematic( F,A, m, P, Q):
    # fnEKF_predict_kinematic implements the extended Kalman Filter predict step for kinematic filtering.
    # F is the nonlinear dynamics function.
    # A is the Jacobian of the function F evaluated at m.
    # m is the mean, P is the covariance matrix.
    # process noise: Q matrix
    m_pred = np.dot(F,m);
    P_pred = np.add(np.dot(np.dot(A,P),np.transpose(A)),Q);
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

def fnEKF_update_kinematic(m_minus, P_minus, y,H,M, R ):
    # fnEKF_update_kinematic implements the extended Kalman Filter update step for kinematic filtering.
    # m_minus,P_minus: state vector and covariance matrix
    # y is the measurement vector. H is the nonlinear measurement function evaluated at m_minus and
    # M is its Jacobian evaluated at m_minus. R is the measurement covariance matrix.
    innovation_mean = H;
    prediction_covariance = np.add(R ,np.dot(M,np.dot(P_minus,np.transpose(M))));
    KalmanGain = np.dot(np.dot(P_minus,np.transpose(M)),np.linalg.inv(prediction_covariance));
    # Calculate estimated mean state vector and its covariance matrix.
    m = m_minus + np.dot(KalmanGain , (y - innovation_mean));
    
    P = np.subtract(P_minus,np.dot(np.dot(KalmanGain,prediction_covariance),np.transpose(KalmanGain)));
    return m,P

def fnEKF_predict_dynamic( F,A, m, P, Q):
    # fnKF_predict implements the extended Kalman Filter predict step. DD-EKF
    # F is the nonlinear discrete-time dynamics function.
    # A is the Jacobian of the function F evaluated at m.
    # m is the mean, P is the covariance matrix.
    # process noise: Q matrix
    m_pred = F(m) ;#np.dot(F,m);
    P_pred = np.add(np.dot(np.dot(A,P),np.transpose(A)),Q);
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred, P_pred

########################################################################################################
def fnCD_EKF(m,P,Xradar,R,t_y,steps,Qc,L,option):
    # arrays to hold estimates.
    x_hat = np.zeros([6,len(t_y)],dtype=np.float64);
    P_hat = np.zeros([6,6,len(t_y)],dtype=np.float64);
    x_hat[:,0] = m;
    P_hat[:,:,0] = P;
    
    # Measurement matrix for update step.
    M = np.zeros([3,6],dtype=np.float64);
    del_y = float(t_y[1]-t_y[0]);
    # --------------------------------------------------------- #
    ## Filtering
    if option == '1':
        for index in range(1,len(t_y)):
            m_pred,P_pred  =fnCD_EKF_predict_MC_RK4( m,P,del_y,steps,Qc,L);
            
            pos = np.array([m_pred[0],m_pred[1],m_pred[2]],dtype=np.float64);
            Htilde = Obs.fnJacobianH(pos);  
            M[:,0] = Htilde[:,0]; M[:,1] =Htilde[:,1];M[:,2] =Htilde[:,2];
            m,P = fnEKF_update_kinematic(m_pred, P_pred, Xradar[:,index],Obs.fnH(pos),M, R );

            x_hat[:,index] = m;
            P_hat[:,:,index]  = P;
    elif option == '2':
        for index in range(1,len(t_y)):
            m_pred,P_pred  =fnCD_EKF_predict_STM_RK4( m,P,del_y,steps,Qc,L);
            pos = np.array([m_pred[0],m_pred[1],m_pred[2]],dtype=np.float64);
            Htilde = Obs.fnJacobianH(pos);  
            M[:,0] = Htilde[:,0]; M[:,1] =Htilde[:,1];M[:,2] =Htilde[:,2];
            m,P = fnEKF_update_kinematic(m_pred, P_pred, Xradar[:,index],Obs.fnH(pos),M, R );

            x_hat[:,index] = m;
            P_hat[:,:,index]  = P;
    elif option  == '3':
        for index in range(1,len(t_y)):
            m_pred, P_pred = fnCD_EKF_predict_D_Euler( m,P,del_y,steps,Qc,L);            
            pos = np.array([m_pred[0],m_pred[1],m_pred[2]],dtype=np.float64);
            Htilde = Obs.fnJacobianH(pos);  
            M[:,0] = Htilde[:,0]; M[:,1] =Htilde[:,1];M[:,2] =Htilde[:,2];
            m,P = fnEKF_update_kinematic(m_pred, P_pred, Xradar[:,index],Obs.fnH(pos),M, R );

            x_hat[:,index] = m;
            P_hat[:,:,index]  = P;
    elif option  == '4':
        for index in range(1,len(t_y)):
            m_pred, P_pred  =fnCD_EKF_predict_D_SRK2( m,P,del_y,steps,Qc,L);
            pos = np.array([m_pred[0],m_pred[1],m_pred[2]],dtype=np.float64);
            Htilde = Obs.fnJacobianH(pos);  
            M[:,0] = Htilde[:,0]; M[:,1] =Htilde[:,1];M[:,2] =Htilde[:,2];
            m,P = fnEKF_update_kinematic(m_pred, P_pred, Xradar[:,index],Obs.fnH(pos),M, R );

            x_hat[:,index] = m;
            P_hat[:,:,index]  = P;
    elif option  == '5':
        for index in range(1,len(t_y)):
            m_pred, P_pred  =fnCD_EKF_predict_D_SRK4( m,P,del_y,steps,Qc,L);
            pos = np.array([m_pred[0],m_pred[1],m_pred[2]],dtype=np.float64);
            Htilde = Obs.fnJacobianH(pos);  
            M[:,0] = Htilde[:,0]; M[:,1] =Htilde[:,1];M[:,2] =Htilde[:,2];
            m,P = fnEKF_update_kinematic(m_pred, P_pred, Xradar[:,index],Obs.fnH(pos),M, R );

            x_hat[:,index] = m;
            P_hat[:,:,index]  = P;
        
    return x_hat,P_hat

########################################################################################################
## Unscented Kalman Filter Functions 
def fnUT_sigmas(X,P,params_vec):
    # Implementation of ut_sigmas.m of the ekfukf toolbox
    #A = np.linalg.cholesky(P);
    A,definite = MathsFn.schol(P); # 21/07/16
##    if definite == -1:
##        print 'Covariance matrix is negative definite.'
##    elif definite == 0:
##        print 'Covariance matrix is positive semidefinite.'
##    else:
##        print 'Covariance matrix is positive definite.'
    n = params_vec[3]; kappa = params_vec[2];
    sigmas = np.vstack((np.zeros_like(X),A ,-A) );
    c  = n + kappa;
    sigmas = np.sqrt(c)*sigmas;
    sigmas  = np.add(sigmas,np.tile(X,(2*n+1,1)));
    return sigmas

def fnUT_weights(params):
    # X - state vector
    # P - covariance matrix
    # params_vec = [alpha,beta,kappa,n]
    # emulates ut_weights.m of the ekfukftoolbox
    alpha = float(params[0]); 
    beta = float(params[1]);
    kappa = float(params[2]); 
    n = params[3];
    lambd = np.square(alpha) * (float(n) + kappa) - float(n);
    Weights_Mean = np.zeros((2*n+1),dtype=np.float64);
    Weights_Cov = np.zeros((2*n+1),dtype=np.float64);
    Weights_Mean[0] = lambd/(float(n)+lambd);
    Weights_Cov[0] = lambd/(float(n)+lambd) + (1-np.square(alpha) + beta);
    for index in range(1,2*n+1):
        Weights_Mean[index] = 1 / (2 * (float(n) + lambd));
        Weights_Cov[index] = Weights_Mean[index];
    return Weights_Mean,Weights_Cov

def fnUKF_predict_kinematic( stm, m, P, Q,params_vec):
    # Form the sigma points of x
    sigmas = fnUT_sigmas(m,P,params_vec);
    # Compute weights
    Wm,Wc = fnUT_weights(params_vec);  
    n = params_vec[3];
    # Propagate sigma points through the (non)linear model.
    yo = np.dot(stm,sigmas[0,:]);
    Y = np.zeros([n,2*n+1],dtype=np.float64); # sigma points of y
    Y[:,0] = yo;
    
    mu = Wm[0]*Y[:,0];
    for index in range(1,2*n+1):
        Y[:,index] = np.dot(stm,sigmas[index,:]);
        mu +=  Wm[index]*Y[:,index];

    Sk  = np.zeros([n,n],dtype=np.float64);
    
    for index in range (0,2*n+1):
        diff = np.subtract(Y[:,index],mu);
        produ = np.multiply.outer(diff,diff); 
        Sk = np.add(Sk,Wc[index]*produ);
        
    m_pred = mu;
    P_pred = np.add(Sk,Q);
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.
    return m_pred,P_pred

def fnUKF_update_kinematic(m_minus, P_minus, y,fnH,R,params_vec):
    # Form the sigma points of x
    sigmas = fnUT_sigmas(m_minus,P_minus,params_vec);
    # Compute weights
    Wm,Wc = fnUT_weights(params_vec);  
    
    n = params_vec[3];
    # Propagate sigma points through the (non)linear model.    
    pos = np.array([sigmas[0,0],sigmas[0,3],sigmas[0,6]],dtype=np.float64);
    yo = fnH(pos);
    Y = np.zeros([np.shape(yo)[0],2*n+1],dtype=np.float64); # sigma points of y
    Y[:,0] = yo;
    
    mu = Wm[0]*Y[:,0];
    for index in range(1,2*n+1):
        pos = np.array([sigmas[index,0],sigmas[index,3],sigmas[index,6]],dtype=np.float64);
        Y[:,index] = fnH(pos);
        mu = Wm[index]*Y[:,index];

    Sk  = np.zeros([np.shape(yo)[0],np.shape(yo)[0]],dtype=np.float64);
    Ck  = np.zeros([n,np.shape(yo)[0]],dtype=np.float64);
    
    for index in range (0,2*n+1):
        diff = np.subtract(Y[:,index],mu);
        produ = np.multiply.outer(diff,diff); 
        Sk = np.add(Sk,Wc[index]*produ); 
        diff1 = np.subtract(sigmas[index,:],m_minus);
        produ1 = np.multiply.outer(diff1,diff);    
        Ck = np.add(Ck,Wc[index]*produ1); 
                
    Sk = np.add(Sk,R);     
    KalmanGain = np.dot(Ck,np.linalg.inv(Sk));

    # Calculate estimated mean state vector and its covariance matrix.
    m = m_minus + np.dot(KalmanGain ,np.subtract(y,mu));
    P = np.subtract(P_minus,np.dot(np.dot(KalmanGain,Sk),np.transpose(KalmanGain)));
    return m,P

## Continuous-Discrete Unscented Kalman Filtering ###################################
def fnMoments_CD_UKF(t,xaug,Q,L):
    # Extract mean state vector and covariance matrix.
    x = xaug[0:6];
    P = np.reshape(xaug[6:],(6,6));
    
    # Define the UT parameters.
    alpha = 1;
    beta = 2;
    kappa = 2;
    n = 6;
    params = np.array([alpha,beta,kappa,n],dtype=int);
    Wm, Wc = fnUT_weights(params);
    # Form the sigma points of x
    sigmas = fnUT_sigmas(x,P,params);

    # Propagate sigma points through the (non)linear model.
    Y = np.zeros([n,2*n+1],dtype=np.float64); # sigma points of y

    for index in range(0,2*n+1):
        Y[:,index] = DynFn.fnKepler_J2(t,sigmas[index,:]);

    mdot = np.dot(Y,Wm);
    
    sigma_matrix = np.tile(np.transpose(Wm),(2*n+1,1));
    
    diff = np.eye(2*n+1,dtype=np.float64) - sigma_matrix;
    Wc_mat = np.diag(Wc);

    W = np.dot(np.dot(diff,Wc_mat),diff.T);

    Pdot = np.dot(np.dot(sigmas.T,W),np.transpose(Y)) + np.dot(np.dot(Y,W),sigmas) + np.dot(np.dot(L,Q),L.T);
    # Ensure symmetry
    Pdot = 0.5*(Pdot+Pdot.T);
    # and positive semidefiniteness.
    Sdot,definite = MathsFn.schol(Pdot);
    if definite != 1:
        print 'Moment propagation made the covariance matrix not positive definite.'
    Pdot = np.dot(Sdot,Sdot.T);
    
    xaug[0:6] = mdot;
    xaug[6:] = np.reshape(Pdot,6*6);
    return xaug

def fnCD_UKF_predict_MC_RK4( m,P,dt,steps,Q,L):
    # fnCD_UKF_predict_MC_RK4 implements the continuous-discrete unscented Kalman Filter predict step.
    # This is the MC-RK4 method: Mean-Covariance Runge-Kutta 4th order
    # m is the mean, P is the covariance matrix.
    # Q is the process noise matrix, L is the dispersion matrix.
    # steps = Number of steps in the RK4 integration
    # dt = time interval from k to k+1.

    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);
    # moment_vector contains the mean vector and covariance matrix which need to be integrated.
    moment_vector = np.zeros((6 + 6*6,len(td)),dtype = np.float64);
    moment_vector[0:6,0] = m; # mean vector
    moment_vector[6:,0] = np.reshape(P,6*6); # covariance matrix 

    for index in range(1,len(td)):
        # Perform RK4 numerical integration on the system of differential equations for the moment.
        moment_vector[:,index] = ni.fnRK4_vector(fnMoments_CD_UKF,dt/steps,moment_vector[:,index-1],td[index-1],Q,L);
        
    # Extract the last mean vector and covariance matrix. These correspond to the mean vector and covariance matrix at time index k+1.
    m_pred = moment_vector[0:6,len(td)-1];
    P_pred = np.reshape(moment_vector[6:,len(td)-1],(6,6));
    # m_pred and P_pred are the predicted mean state vector and covariance
    # matrix at the current time step before seeing the measurement.

    # Ensure symmetry
    P_pred = 0.5*(P_pred+P_pred.T);
    # and positive semidefiniteness.
    S_pred,definite = MathsFn.schol(P_pred);
    if definite != 1:
        print 'Moment propagation made the covariance matrix not positive definite.'
    P_pred = np.dot(S_pred,S_pred.T);
    return m_pred, P_pred

def fnUKF_update_dynamic(m_minus, P_minus, y,fnH,R,params_vec):
    # Form the sigma points of x
    sigmas = fnUT_sigmas(m_minus,P_minus,params_vec);
    # Compute weights
    Wm,Wc = fnUT_weights(params_vec);  
    
    n = params_vec[3];
    # Propagate sigma points through the (non)linear model.    
    pos = np.array([sigmas[0,0],sigmas[0,1],sigmas[0,2]],dtype=np.float64);
    yo = fnH(pos);
    Y = np.zeros([np.shape(yo)[0],2*n+1],dtype=np.float64); # sigma points of y
    Y[:,0] = yo;
    
    mu = Wm[0]*Y[:,0];
    for index in range(1,2*n+1):
        pos = np.array([sigmas[index,0],sigmas[index,1],sigmas[index,2]],dtype=np.float64);
        Y[:,index] = fnH(pos);
        mu = Wm[index]*Y[:,index];

    Sk  = np.zeros([np.shape(yo)[0],np.shape(yo)[0]],dtype=np.float64);
    Ck  = np.zeros([n,np.shape(yo)[0]],dtype=np.float64);
    
    for index in range (0,2*n+1):
        diff = np.subtract(Y[:,index],mu);
        produ = np.multiply.outer(diff,diff); 
        Sk = np.add(Sk,Wc[index]*produ); 
        diff1 = np.subtract(sigmas[index,:],m_minus);
        produ1 = np.multiply.outer(diff1,diff);    
        Ck = np.add(Ck,Wc[index]*produ1); 
                
    Sk = np.add(Sk,R);     
    KalmanGain = np.dot(Ck,np.linalg.inv(Sk));

    # Calculate estimated mean state vector and its covariance matrix.
    m = m_minus + np.dot(KalmanGain ,np.subtract(y,mu));
    P = np.subtract(P_minus,np.dot(np.dot(KalmanGain,Sk),np.transpose(KalmanGain)));

    # Ensure symmetry
    P = 0.5*(P+P.T);
    # and positive semidefiniteness.
    S,definite = MathsFn.schol(P);
    P = np.dot(S,S.T);
    return m,P

def fnCD_UKF_predict_M(m,P,dt,steps,L,true_Qc,params):
    ## fn_CD_UKF_predict_M implements the CD-UKF predict steps by integrating the sigma points through the nonlinear dynamics using the RK4 technique.
    ## This function adheres to the Moment Matching approach. Refer to Crouse 2015.
    """@INPROCEEDINGS{6875583, 
    author={D. F. Crouse}, 
    booktitle={2014 IEEE Radar Conference}, 
    title={Cubature Kalman filters for continuous-time dynamic models Part II: A solution based on moment matching}, 
    year={2014}, 
    pages={0194-0199}, 
    month={May}
    }"""
    ## Created: 22 July 2016
    
    # Form the sigma points of x
    sigmas = fnUT_sigmas(m,P,params);
    
    # Compute weights
    Wm,Wc = fnUT_weights(params);

    # time vector td to find the propagation of moment_vector from k to k+1
    td = np.arange(0,dt+dt/steps,dt/steps,dtype=np.float64);
    
    # mean_vector contains the mean vector which needs to be integrated.
    mean_vector = np.zeros((params[3],2*params[3]+1,len(td)),dtype = np.float64);
    mean_vector[:,:,0] = sigmas.T; # mean vector
    
    for jindex in range(0,2*params[3]+1):
        for index in range(1,len(td)):
            # Perform RK4 numerical integration on the differential equation for the mean.
            mean_vector[:,jindex,index] = ni.fnRK4_vector(DynFn.fnKepler_J2,dt/steps,mean_vector[:,jindex,index-1],td[index-1]);
    ## Note: 22/07/16:
    ## The variable mean_vector contains the sigma points to be integrated.
    ## The sigma points are integrated individually by the RK4 function.
        
    # Extract the last mean vector. This corresponds to the mean vector at time index k+1.
    m_pred = np.dot(mean_vector[:,:,len(td)-1],Wm);

    Sk  = np.zeros([params[3],params[3]],dtype=np.float64);
    
    for index in range (0,2*params[3]+1):
        diff = np.subtract(mean_vector[:,index,len(td)-1],m_pred);
        produ = np.multiply.outer(diff,diff); 
        Sk = np.add(Sk,Wc[index]*produ);
        
    P_pred = Sk + np.dot(np.dot(L,true_Qc),L.T);
    return m_pred, P_pred
######################################################################################################################################
## Gauss-Newton functions
def fnMVA( Lambda, TotalObservationMatrixTranspose, RynInv, Ryn ,delta_Y ):
    # FNMVA implements the Minimum Variance Algorithm (MVA).
    # Estimated covariance matrix
    #S_hat = np.linalg.inv(Lambda);
    # Filter matrix W in TFE == matrix M in Tapley,Schutz,Born.
    #W = np.dot(np.dot(S_hat,TotalObservationMatrixTranspose),RynInv);
    A = Lambda;
    b = np.dot(TotalObservationMatrixTranspose,RynInv);
    W = np.linalg.solve(A,b);
    S_hat = np.dot(np.dot(W,Ryn),np.transpose(W));
    # Estimated perturbation vector
    delta_X_hat = np.dot(W,delta_Y);
    return delta_X_hat,S_hat

def fnGaussNewtonBatch(Xdash,timevec,L,M,Xradar,RynInv,Ryn):
    # From initial estimate of state vector xdash, generate a nominal trajectory over L (the filter window length)
    Xnom = DynFn.fnGenerate_Nominal_Trajectory(Xdash,timevec[0:L]);
    # Build total observation matrix
    # Total Observation Matrix
    TotalObservationMatrix = np.zeros([3*L,6],dtype=np.float64);
    TotalObservationMatrixTranspose = np.zeros([6,3*L],dtype=np.float64); 

    # Initialize the matrices
    M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,0] );
    Phi = np.reshape(Xnom[6:,0],(6,6));
    TotalObservationMatrix[0:3,:] = np.dot(M,Phi); 
    TotalObservationMatrixTranspose[:,0:3] = np.transpose(np.dot(M,Phi));
    
    TotalObservationVector = np.zeros([3*L],dtype=np.float64);
    # Load the sensor input into the TotalObservationVector at the validity instant Tvi == tstart
    TotalObservationVector[0:3] =  Xradar[0:3,0];

    # delta Y vector == simulated perturbation vector in TFE
    delta_Y = np.zeros([3*L],dtype=np.float64);

    # Load the starting sample into the simulated perturbation vector
    delta_Y[0:3] = np.transpose(TotalObservationVector[0:3] - Obs.fnH( Xnom[0:3,0]) ); 

    # Load the filter's input stack
    for index in range(1,L):# tstart+1:tstop
        M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,index] );
        Phi = np.reshape(Xnom[6:,index],(6,6));
        # The T matrix and its transpose
        TotalObservationMatrix[3*index:3*index+3,:] = np.dot(M,Phi);
        TotalObservationMatrixTranspose[:,3*index:3*index+3] = np.transpose(np.dot(M,Phi));

        TotalObservationVector[3*index:3*index+3] =  Xradar[0:3,index];
        delta_Y[3*index:3*index+3] = np.transpose(TotalObservationVector[3*index:3*index+3] - Obs.fnH( Xnom[0:3,index]) );

    # Fisher information matrix
    Lambda = np.dot(np.dot(TotalObservationMatrixTranspose,RynInv),TotalObservationMatrix);
    delta_X_hat,S_hat = fnMVA( Lambda, TotalObservationMatrixTranspose, RynInv, Ryn,delta_Y )
    Xdash = Xnom[0:6,L-1] + delta_X_hat;

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = np.dot(np.transpose(delta_Y),np.dot(RynInv,delta_Y));
    
    return Xdash,S_hat,Ji
# ----------------------------------------------------------------------------------- #
# Gauss-Newton filtering in Fixed Memory Length form
def fnGaussNewtonFilter(Xdash,timevec,L,M,Xradar,Rinv,R):
    # From initial estimate of state vector xdash, generate a nominal trajectory over L (the filter window length)
    Xnom = DynFn.fnGenerate_Nominal_Trajectory(Xdash,timevec);
    # Build total observation matrix
    # Total Observation Matrix
    TotalObservationMatrix = np.zeros([3*L,6],dtype=np.float64);
    TotalObservationMatrixTranspose = np.zeros([6,3*L],dtype=np.float64); 

    # Initialize the matrices
    M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,0] );
    Phi = np.reshape(Xnom[6:,0],(6,6));
    TotalObservationMatrix[0:3,:] = np.dot(M,Phi); 
    TotalObservationMatrixTranspose[:,0:3] = np.transpose(np.dot(M,Phi));
    
    TotalObservationVector = np.zeros([3*L],dtype=np.float64);
    # Load the sensor input into the TotalObservationVector at the validity instant Tvi == tstart
    TotalObservationVector[0:3] =  Xradar[0:3,0];

    # delta Y vector == simulated perturbation vector in TFE
    delta_Y = np.zeros([3*L],dtype=np.float64);

    # Load the starting sample into the simulated perturbation vector
    delta_Y[0:3] = np.transpose(TotalObservationVector[0:3] - Obs.fnH( Xnom[0:3,0]) ); 

    X_hat = np.zeros([6,len(timevec)],dtype=np.float64);
    S_hat = np.zeros([6,6,len(timevec)],dtype=np.float64);
    
    # Load the filter's input stack
    for index in range(1,len(timevec)):
        M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,index] );
        Phi = np.reshape(Xnom[6:,index],(6,6));

        if index < L:    
            # The T matrix and its transpose
            TotalObservationMatrix[3*index:3*index+3,:] = np.dot(M,Phi);
            TotalObservationMatrixTranspose[:,3*index:3*index+3] = np.transpose(np.dot(M,Phi));

            TotalObservationVector[3*index:3*index+3] =  Xradar[0:3,index];
            delta_Y[3*index:3*index+3] = np.transpose(TotalObservationVector[3*index:3*index+3] - Obs.fnH( Xnom[0:3,index]) );

            # Fisher information matrix
            RynInv = DynFn.fn_Create_Concatenated_Block_Diag_Matrix(Rinv,index);
            Ryn = DynFn.fn_Create_Concatenated_Block_Diag_Matrix(R,index);
            Lambda = np.dot(np.dot(TotalObservationMatrixTranspose[:,0:3*index+3],RynInv),TotalObservationMatrix[0:3*index+3,:]);
            delta_X_hat,S_hat[:,:,index] = fnMVA( Lambda, TotalObservationMatrixTranspose[:,0:3*index+3], RynInv, Ryn,delta_Y[0:3*index+3] )
            X_hat[:,index] = Xnom[0:6,index] + delta_X_hat;
            
        else:
            # Forget outdated data
            # Perform circular shift to get rid of old data.
            print 'Cycling the filter'
            
            #M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,index] );
            #Phi = np.reshape(Xnom[6:,index],(6,6));
            # The T matrix and its transpose
            # Replace the oldest sample by new data
            TotalObservationMatrix[3*0:3*0+3,:] = np.dot(M,Phi);
            TotalObservationMatrixTranspose[:,3*0:3*0+3] = np.transpose(np.dot(M,Phi));

            TotalObservationVector[3*0:3*0+3] =  Xradar[0:3,index];
            delta_Y[3*0:3*0+3] = np.transpose(TotalObservationVector[3*0:3*0+3] - Obs.fnH( Xnom[0:3,index]) );

            # and reset the matrices.
            TotalObservationMatrix = np.roll(TotalObservationMatrix,-3,axis=0);
            TotalObservationMatrixTranspose = np.roll(TotalObservationMatrixTranspose,-3,axis=1);

            TotalObservationVector = np.roll(TotalObservationVector,-3);
            delta_Y = np.roll(delta_Y,-3);

            Lambda = np.dot(np.dot(TotalObservationMatrixTranspose,RynInv),TotalObservationMatrix);
            delta_X_hat,S_hat[:,:,index] = fnMVA( Lambda, TotalObservationMatrixTranspose, RynInv, Ryn,delta_Y );
            X_hat[:,index] = Xnom[0:6,index] + delta_X_hat;

    # Cost Ji of cost function. (Crassidis pg 28)
    #Ji = np.dot(np.transpose(delta_Y),np.dot(RynInv,delta_Y));
    
    return X_hat,S_hat
