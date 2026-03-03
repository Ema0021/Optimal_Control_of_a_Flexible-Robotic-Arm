import numpy as np
import sympy as sp
from sympy import Matrix 

import flex_arm_parameters as params 

print_syb = False
print_num = False

theta1, theta2, theta1_dot, theta2_dot, u, dt = sp.symbols('theta1 theta2 theta1_dot theta2_dot u dt')

# System parameters
m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2 = sp.symbols('m1 m2 l1 l2 r1 r2 I1 I2 g f1 f2')

# Parameters for the Flexible arm dynamics
numerical_value = params.numerical_value = {

    'm1': 2,            # mass arm 1      
    'm2': 2,            # mass arm 2
    'l1': 1.5,          # length arm 1
    'l2': 1.5,          # length arm 2
    'r1': 0.75,         # distances between the pivot points of the link and their center of mass 1
    'r2': 0.75,         # distances between the pivot points of the link and their center of mass 2
    'I1': 1.5,          # inertia 1
    'I2': 1.5,          # inertia 2
    'g': 9.81,          # gravitational acceleration [m/s^2]
    'f1': 0.1,          # viscous friction coefficients 1
    'f2': 0.1,          # viscous friction coefficients 2
    'dt': params.dt,    # discretization step
    'ns': params.ns,    # number of states
    'nu': params.nu,    # number of inputs   
    
}


def M_matrix(theta2):
    # Inertia matrix M
    M11 = I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) + 2 * m2 * l1 * r2 * sp.cos(theta2)
    M12 = I2 + m2 * r2**2 + m2 * l1 * r2 * sp.cos(theta2)
    M21 = M12
    M22 = I2 + m2 * r2**2
    M = sp.Matrix([[M11, M12], [M21, M22]])

    return M

def C_matrix(theta1, theta2, theta1_dot, theta2_dot): 
     # Coriolis matrix C
    C11 = -m2 * l1 * r2 * theta2_dot * sp.sin(theta2) * (theta2_dot + 2 * theta1_dot)
    C21 = m2 * l1 * r2 * sp.sin(theta2) * (theta1_dot**2)
    C = Matrix([C11, C21])

    return C

def G_matrix(theta1, theta2):
    # Gravitational acceleration matrix G
    G11 = g * (m1 * r1 + m2 * l1) * sp.sin(theta1) + g * m2 * r2 * sp.sin(theta1 + theta2)
    G21 = g * m2 * r2 * sp.sin(theta1 + theta2)
    G = sp.Matrix([G11, G21])

    return G

def F_matrix(f1, f2):
    # Viscous friction matrix F
    F11 = f1
    F12 = 0
    F21 = 0
    F22 = f2
    F = sp.Matrix([[F11, F12], [F21, F22]])

    return F



def symbolic_dynamic():
    """
        Compute the dynamics of a Flexible Robotic Arm keeping the states in a symbolic way while the parameters are substituted with their numerical values.

        Parameters:
            - m1, m2: masses of the links.
            - l1, l2: lengths of the links.
            - r1, r2: distances between the pivot points of the link and their center of mass.
            - I1, I2: inertia.
            - g: gravitational acceleration.
            - f1, f2: viscous friction coefficients.

        Returns:
            - xx_plus_symb: symbolic x_{t+1}, state at t+1: [theta1_plus, theta2_plus, theta1_dot_plus, theta2_dot_plus].
    """

    # Unpacking Matrix M
    M = M_matrix(theta2)
    M11 =  M[0,0]
    M12 =  M[0,1]
    M21 =  M[1,0]
    M22 =  M[1,1]


    # Unpacking Matrix of Coriolis C
    C = C_matrix(theta1, theta2, theta1_dot, theta2_dot)
    C11 = C[0]
    C21 = C[1]


    # Unpacking Gravity Matrix G
    G = G_matrix(theta1, theta2)
    G11 = G[0]
    G21 = G[1]


    # Unpacking Matrix of Viscous Friction F
    F = F_matrix(f1, f2)
    F11 = F[0,0]
    F12 = F[0,1]
    F21 = F[1,0]
    F22 = F[1,1]


    
    #############################
    # DYNAMIC EQUATIONS         #
    #############################

    # x3_dot from dynamic equations 
    theta1_dotdot = (M22/(M22*M11-M12**2) * (u + (M12/M22)*(C21+F22*theta2_dot+G21) - (C11+F11*theta1_dot+G11) )).subs(numerical_value)

    # x4_dot from dynamic equations 
    theta2_dotdot = (-1/M22 *( (M12*theta1_dotdot) + C21 + F22*theta2_dot + G21 )).subs(numerical_value)

    # Dynamics equations
    theta1_plus = theta1 + params.dt * (theta1_dot)
    theta2_plus = theta2 + params.dt * (theta2_dot)
    theta1_dot_plus = theta1_dot + params.dt * (theta1_dotdot)
    theta2_dot_plus = theta2_dot + params.dt * (theta2_dotdot)

    if print_syb == True:
        print("theta1_plus", theta1_plus)
        print("theta2_plus", theta2_plus)
        print("theta1_dot_plus", theta1_dot_plus)
        print("theta2_dot_plus", theta2_dot_plus)
    
    xx_plus_symb = sp.Matrix([theta1_plus, theta2_plus, theta1_dot_plus, theta2_dot_plus])


    #############################
    # GRADIENT OF THE DYNAMIC   #
    #############################

    # Gradient of f1 wrt the state variables
    fx11 = sp.diff(theta1_plus,theta1)
    fx21 = sp.diff(theta1_plus,theta2)
    fx31 = sp.diff(theta1_plus,theta1_dot)
    fx41 = sp.diff(theta1_plus,theta2_dot)

    # Gradient of f2 wrt the state variables
    fx12 = sp.diff(theta2_plus,theta1)
    fx22 = sp.diff(theta2_plus,theta2)
    fx32 = sp.diff(theta2_plus,theta1_dot)
    fx42 = sp.diff(theta2_plus,theta2_dot) 

    # Gradient of f3 wrt the state variables
    fx13 = sp.diff(theta1_dot_plus,theta1)
    fx23 = sp.diff(theta1_dot_plus,theta2)
    fx33 = sp.diff(theta1_dot_plus,theta1_dot)
    fx43 = sp.diff(theta1_dot_plus,theta2_dot)

    # Gradient of f4 wrt the state variables
    fx14 = sp.diff(theta2_dot_plus,theta1)
    fx24 = sp.diff(theta2_dot_plus,theta2)
    fx34 = sp.diff(theta2_dot_plus,theta1_dot)
    fx44 = sp.diff(theta2_dot_plus,theta2_dot)

    # Gradient of f wrt the input variables
    fu11 = sp.diff(theta1_plus,u)
    fu12 = sp.diff(theta2_plus,u)
    fu13 = sp.diff(theta1_dot_plus,u)
    fu14 = sp.diff(theta2_dot_plus,u)

    # Gradient matrix of f wrt x
    fx_symb= sp.Matrix([
        [fx11, fx12, fx13, fx14],
        [fx21, fx22, fx23, fx24],
        [fx31, fx32, fx33, fx34],
        [fx41, fx42, fx43, fx44]])
    
    
    # Gradient matrix of f wrt u
    fu_symb = sp.Matrix([[fu11, fu12, fu13, fu14]])

    # print("\n fx_symb - Da gradient \n ", fx_symb)
    # print("\n fu_symb - Da gradient  \n ", fu_symb)
    
    return xx_plus_symb, fx_symb, fu_symb


def dynamics(xx, uu, _flag_print=False):
    """
    Compute the next state xx_plus of the robotic arm, given current state xx and input uu.

    Parameters:
        xx: current state vector (theta1, theta2, theta1_dot, theta2_dot) - shape (4,)
        uu: input (torque) - shape (1,)
        _flag_print: whether to print intermediate results (for debugging)

    Returns:
        xx_plus: next state vector (theta1_plus, theta2_plus, theta1_dot_plus, theta2_dot_plus)
    """

    theta1_val, theta2_val, theta1_dot_val, theta2_dot_val = xx
    u_val = uu[0]

    xx_plus = f_dynamics(theta1_val, theta2_val, theta1_dot_val, theta2_dot_val, u_val)
    fx = f_fx(theta1_val, theta2_val, theta1_dot_val, theta2_dot_val, u_val)
    fu = f_fu(theta1_val, theta2_val, theta1_dot_val, theta2_dot_val, u_val)
    
    if _flag_print:
        print("xx =", xx)
        print("uu =", uu)
        print("xx_plus =", xx_plus)
        print("fx =", fx)
        print("fu =", fu)

    return np.array(xx_plus).astype(np.float64).flatten(), \
           np.array(fx).astype(np.float64), \
           np.array(fu).astype(np.float64)



xx_plus_symb, fx_symb, fu_symb = symbolic_dynamic()

# Convert the symbolic dynamics to a numerical function

f_dynamics = sp.lambdify([theta1, theta2, theta1_dot, theta2_dot, u], xx_plus_symb, modules='numpy')

# Anotheer way to compute the dynamics is to use the Jacobian of the symbolic dynamics
# fx_symb = sp.Matrix(xx_plus_symb).jacobian([theta1, theta2, theta1_dot, theta2_dot])
# fu_symb = sp.Matrix(xx_plus_symb).jacobian([u])

# Convert the symbolic gradients to numerical functions
f_fx = sp.lambdify([theta1, theta2, theta1_dot, theta2_dot, u], fx_symb, modules='numpy')
f_fu = sp.lambdify([theta1, theta2, theta1_dot, theta2_dot, u], fu_symb, modules='numpy')