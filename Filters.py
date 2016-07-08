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
# ------------------------- #
# Theoretical background:
# 1. Tracking Filter Engineering, Norman Morrison. 2013
# 2. Statistical Orbit Determination, Tapley, Schutz, Born. 2004
# 3. Bayesian Signal Processing. Sarkka. 2013
# ------------------------- #
# Import libraries
import numpy as np
from numpy import linalg
#import scipy.linalg
# ---------------------------------------- #


########################################################################################################
def fnEKF_predict_kinematic( F,A, m, P, Q):
    # fnKF_predict implements the extended Kalman Filter predict step.
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
    # m_minus,P_minus: state vector and covariance matrix
    # y is the measurement vector. H is the nonlinear measurement function and
    # M is its Jacobian. R is the measurement covariance matrix.
    innovation_mean = H;
    prediction_covariance = np.add(R ,np.dot(M,np.dot(P_minus,np.transpose(M))));
    KalmanGain = np.dot(np.dot(P_minus,np.transpose(M)),np.linalg.inv(prediction_covariance));
    # Calculate estimated mean state vector and its covariance matrix.
    m = m_minus + np.dot(KalmanGain , (y - innovation_mean));
    
    P = np.subtract(P_minus,np.dot(np.dot(KalmanGain,prediction_covariance),np.transpose(KalmanGain)));
    return m,P
########################################################################################################
## Unscented Kalman Filter Functions 
def fnUT_sigmas(X,P,params_vec):
    # Implementation of ut_sigmas.m of the ekfukf toolbox
    A = np.linalg.cholesky(P);
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

    Sk  = np.zeros([n,n],dtype=float);
    
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

    Sk  = np.zeros([np.shape(yo)[0],np.shape(yo)[0]],dtype=float);
    Ck  = np.zeros([n,np.shape(yo)[0]],dtype=float);
    
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

def fnGaussNewtonBatch(Xdash,timevec,L,M,Xradar,RynInv,Ryn,fnGenerate_Nominal_Trajectory,fnH,fnJacobianH,fnMVA):
    # From initial estimate of state vector xdash, generate a nominal trajectory over L (the filter window length)
    Xnom = fnGenerate_Nominal_Trajectory(Xdash,timevec[0:L]);
    # Build total observation matrix
    # Total Observation Matrix
    TotalObservationMatrix = np.zeros([3*L,6],dtype=np.float64);
    TotalObservationMatrixTranspose = np.zeros([6,3*L],dtype=np.float64); 

    # Initialize the matrices
    M[0:3,0:3] = fnJacobianH( Xnom[0:3,0] );
    Phi = np.reshape(Xnom[6:,0],(6,6));
    TotalObservationMatrix[0:3,:] = np.dot(M,Phi); 
    TotalObservationMatrixTranspose[:,0:3] = np.transpose(np.dot(M,Phi));
    
    TotalObservationVector = np.zeros([3*L],dtype=np.float64);
    # Load the sensor input into the TotalObservationVector at the validity instant Tvi == tstart
    TotalObservationVector[0:3] =  Xradar[0:3,0];

    # delta Y vector == simulated perturbation vector in TFE
    delta_Y = np.zeros([3*L],dtype=np.float64);

    # Load the starting sample into the simulated perturbation vector
    delta_Y[0:3] = np.transpose(TotalObservationVector[0:3] - fnH( Xnom[0:3,0]) ); 

    # Load the filter's input stack
    for index in range(1,L):# tstart+1:tstop
        M[0:3,0:3] = fnJacobianH( Xnom[0:3,index] );
        Phi = np.reshape(Xnom[6:,index],(6,6));
        # The T matrix and its transpose
        TotalObservationMatrix[3*index:3*index+3,:] = np.dot(M,Phi);
        TotalObservationMatrixTranspose[:,3*index:3*index+3] = np.transpose(np.dot(M,Phi));

        TotalObservationVector[3*index:3*index+3] =  Xradar[0:3,index];
        delta_Y[3*index:3*index+3] = np.transpose(TotalObservationVector[3*index:3*index+3] - fnH( Xnom[0:3,index]) );

    # Fisher information matrix
    Lambda = np.dot(np.dot(TotalObservationMatrixTranspose,RynInv),TotalObservationMatrix);
    delta_X_hat,S_hat = fnMVA( Lambda, TotalObservationMatrixTranspose, RynInv, Ryn,delta_Y )
    Xdash = Xnom[0:6,L-1] + delta_X_hat;

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = np.dot(np.transpose(delta_Y),np.dot(RynInv,delta_Y));
    
    return Xdash,S_hat,Ji
