import numpy as np
import control as ctrl  # control package python
import scipy as sp

import flex_arm_parameters as params
import flex_arm_compute_equilibria as compeq
import flex_arm_dynamics as dyn


# Number of states and inputs
ns = params.ns
nu = params.nu

xx_eq_2 = params.xx_eq_2
uu_eq_2 = compeq.compute_equilibria(xx_eq_2)
dfx, dfu = dyn.dynamics(xx_eq_2, uu_eq_2)[1:]
AA = dfx.T
BB = dfu.T

# Weights for the Stage Cost
QQt = np.diag([100.0, 100.0, 0.1, 0.1])               # weight for state of stage cost
RRt = 0.01*np.eye(nu)                                 # weight for input of stage cost
QQT = sp.linalg.solve_discrete_are(AA, BB, QQt, RRt)  # ctrl.dare(AA,BB,QQt,RRt)[0]  # weight for the terminal cost 

# Weights for the LQR Trajectory Tracking
QQt_lqr = np.diag([100.0, 100.0, 1.0, 1.0])           # weight for state of stage cost
RRt_lqr = 0.01*np.eye(nu)                             # weight for input of stage cost
QQT_lqr = ctrl.dare(AA,BB,QQt_lqr,RRt_lqr)[0]         # sp.linalg.solve_discrete_are(AA, BB, QQt_lqr, RRt_lqr) # weight for the terminal cost

# Weights for the MPC
QQt_mpc = 0.1*np.diag([100.0, 100.0, 100.0, 30.0])    # weight for state of stage cost
RRt_mpc = 0.01*np.eye(nu)                             # weight for input of stage cost
QQT_mpc = sp.linalg.solve_discrete_are(AA, BB, QQt_mpc, RRt_mpc)  # weight for the terminal cost


# Verifay type and converison 
AA = np.array(AA, dtype=float)
BB = np.array(BB, dtype=float)
QQt = np.array(QQt, dtype=float)
RRt = np.array(RRt, dtype=float)


def stagecost (xx, uu, xx_ref, uu_ref):
  """
  Compute the Stage Cost l(x) and its gradients with respect to xx and uu.

  Parameters:
    - xx: State vector. in R^4*TT
    - uu: Input vector. in R^TT
    - xx_ref: Reference State Vector. in R^4*TT
    - uu_ref: Reference Input Vector. in R^TT

  Returns:
    A tuple containing 
    - ll:  the stage cost 
    - dlx: gradient with respect to xx
    - dlu: gradient with respect to uu
  """
  
  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  # Stage Cost 
  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  # Gradient of the Stage Cost with respect to xx and uu
  dlx = QQt@(xx - xx_ref)  #q_t
  dlu = RRt@(uu - uu_ref)  #r_t

  return ll.squeeze(), dlx.squeeze(), dlu.squeeze()

def terminalcost (xT, xT_ref, QQT):
  """
  Terminal-cost
  Quadratic cost function ll_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

  Parameters:
    - xT: state at time t in R^4 
    - xT_ref: state reference at time t in R^4 

  Return: 
    - llT: the terminal cost
    - dlTx: gradient of the terminal cost with respect to xT

  """
  xT = xT[:,None]
  xT_ref = xT_ref[:,None]

  llT = 0.5*(xT - xT_ref).T@QQT@(xT - xT_ref) # cost at xT,uu

  dlTx = QQT@(xT - xT_ref) # gradient of l wrt x, at xT

  return llT.squeeze(), dlTx.squeeze()
