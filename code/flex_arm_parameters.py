import numpy as np

################################
# BOOLEAND FLAGS               #
################################

# Boolean flag to plot the reference trajectory
flag_plot_ref = True                 # set to True to print the reference trajectory
state_variable = True                # set to True to plot the state variables
flag_plot_armijo = True              # set to True to plot the Armijo descent
flag_plot_descent = True             # set to True to plot the descent
flag_anim = True                     # set to True to animate the pendulum
flag_error_ref = True                # set to True to plot the error between reference and optimal trajectory
flag_plot_vel = True                 # set to True to plot the velocity of the pendulum

armijo_loop = True                   # set to True to print debug information for the Armijo loop TRUE means that the Armijo closed loop 
                                     # oterwise FALSE means that the Armijo open loop

print_debug = False                  # set to True to print debug information

simulation_review = False            # set to True to load the simulation data from a file

flag_MPC_difference_constr = True    # set to True to see the difference between MPC with constraints on input and without it

#################################
# PARAMETERS FOR THE SIMULATION #
#################################

dt = 1e-2               # discretization step
ns = 4                  # number of states
nu = 1                  # number of inputs
tf = 10                 # simulation time
TT = int(tf / dt)       # time samples
maxiters = 5            # iterations for Newton's method 


#################################
# PARAMETERS FOR THE EQUILIBRIA #
#################################

# 1st Equilibrium State
theta1_eq1 = np.radians(210.0)  #[rad]
theta2_eq1 = - theta1_eq1       #[rad]
theta1_dot_eq1 = 0              #[rad/s]
theta2_dot_eq1 = 0              #[rad/s]

xx_eq_1 = [theta1_eq1, theta2_eq1, theta1_dot_eq1, theta2_dot_eq1]


# 2nd Equilibrium state
theta1_eq2 = np.radians(30.0)   #[rad]
theta2_eq2 = - theta1_eq2       #[rad]
theta1_dot_eq2 = 0              #[rad/s]
theta2_dot_eq2 = 0              #[rad/s]


xx_eq_2 = [theta1_eq2, theta2_eq2, theta1_dot_eq2, theta2_dot_eq2]


# This is the 3rd and 4th equilibrium states, which are not used in the simulation but can be useful for analysis.

# Frist  State Equilibrium
# 1st Equilibrium State
theta1_eq3 = np.radians(0.0)    #[rad]
theta2_eq3 = - theta1_eq3       #[rad]
theta1_dot_eq3 = 0              #[rad/s]
theta2_dot_eq3 = 0              #[rad/s]

xx_eq_3 = [theta1_eq3, theta2_eq3, theta1_dot_eq3, theta2_dot_eq3]


# 2nd Equilibrium state
theta1_eq4 = np.radians(45.0)   #[rad]
theta2_eq4 = - theta1_eq4       #[rad]
theta1_dot_eq4 = 0              #[rad/s]
theta2_dot_eq4 = 0              #[rad/s]


xx_eq_4 = [theta1_eq4, theta2_eq4, theta1_dot_eq4, theta2_dot_eq4]



######################################
# MPC TIME PARAMETERS AND CONSTRAINTS#
######################################
T_pred_mpc = 20                 # prediction horizon for the MPC
T_sim_mpc = TT                  # simulation time for the MPC, it is equal to the total number of time samples

# Constraints for the MPC
x1_max = 10                     # maximum value for x1
x1_min = -x1_max                # minimum value for x1


x2_max = 10                     # maximum value for x2
x2_min = -x2_max                # minimum value for x2


x1_vel = 10                     # maximum value for x1
x1_vel = -x1_vel                # minimum value for x1


x2_vel = 10                     # maximum value for x2
x2_min = -x2_vel                # minimum value for x2


u1_max = 700                    # minimum value for u1
u1_min = -u1_max                # maximum value for u1


################################
# X0 PERTURBATIONS FOR LQR     #  
################################


perturbation_LQR_small = np.array([0.05, -0.03, 0.0, 0.0])
perturbation_LQR_medium = np.array([0.1, -0.2, 0.1, 0.1])
perturbation_LQR_large = np.array([0.5, -0.3, 0.0, 0.0])

# perturbation_MPC_small = np.array([0.05, -0.05, 0.02, 0.02])  
# perturbation_MPC_medium = np.array([0.1, -0.2, 0.1, 0.1])  
# perturbation_MPC_large = np.array([0.5, -0.3, 0.3, 0.3])  

# This pertubation is used to test the MPC for comparison with the LQR
perturbation_MPC_small = np.array([0.05, -0.03, 0.0, 0.0])
perturbation_MPC_medium = np.array([0.1, -0.2, 0.1, 0.1])
perturbation_MPC_large = np.array([0.5, -0.3, 0.0, 0.0])


