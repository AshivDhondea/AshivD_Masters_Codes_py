## Num_Integ.py
# ------------------------- #
# Description:
# A collection of Numerical Methods functions.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits:
# ------------------------- #

def fnRK4_vector(f, dt, x,t):
    # Execute one RK4 integration step
    k1 = dt*  f(t         ,x         );
    k2 = dt*  f(t + 0.5*dt,x + 0.5*k1);
    k3 = dt*  f(t + 0.5*dt,x + 0.5*k2);
    k4 = dt*  f(t +     dt,x +     k3);
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
