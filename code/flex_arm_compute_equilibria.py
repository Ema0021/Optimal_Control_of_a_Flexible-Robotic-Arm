import numpy as np
from scipy.optimize import fsolve
import flex_arm_dynamics as dyn


def compute_equilibria(xx_eq):
    """
       Given equilibrium state, this function computes the associated equilibrium input.

        Parameters:
           - xx_eq: State vector [theta1, theta2, theta1_dot, theta2_dot].

        Returns:
           - u_eq: Equilibrium input (torque) that keeps the system in equilibrium.
    """
    
    # Unpacking the state variable
    theta1 = xx_eq[0]       #[rad]
    theta2 = xx_eq[1]        #[rad]
    theta1_dot = xx_eq[2]   #[rad/s]
    theta2_dot = xx_eq[3]   #[rad/s]

    # Unpacking Matrix of Inertia M
    M = dyn.M_matrix(theta2)
    M = M.subs(dyn.numerical_value)
    M11 = M[0,0]
    M12 = M[0,1]
    M21 = M[1,0]
    M22 = M[1,1]


    # Unpacking Matrix of Coriolis C
    C = dyn.C_matrix(theta1, theta2, theta1_dot, theta2_dot)
    C = C.subs(dyn.numerical_value)
    C11 = C[0]
    C21 = C[1]


    # Unpacking Gravity Matrix G
    G = dyn.G_matrix(theta1, theta2)
    G = G.subs(dyn.numerical_value)
    G11 = G[0]
    G21 = G[1]


    # Unpacking Matrix of Viscous Friction F
    F = dyn.F_matrix(dyn.numerical_value['f1'], dyn.numerical_value['f2'])
    F = F.subs(dyn.numerical_value)
    F11 = F[0,0]
    F12 = F[0,1]
    F21 = F[1,0]
    F22 = F[1,1]

    u_init=1

    def equilibria_equation(u):
        return  float( M22/(M22*M11-M12**2) * (u + (M12/M22)*(C21+F22*theta2_dot+G21) - (C11+F11*theta1_dot+G11) ) )
        
    u_eq = fsolve(equilibria_equation, u_init)
    return u_eq

