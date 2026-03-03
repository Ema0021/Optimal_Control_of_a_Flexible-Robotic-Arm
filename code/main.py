"""
    OPTIMAL CONTROL OF A DOUBLE PENDULUM
    Authors: Emanuele Monsellato
    Course: Optimal Control 2024/2025

"""

import numpy as np
import os
from termcolor import colored
import sys
import flex_arm_parameters as params
import flex_arm_compute_equilibria as compeq
import flex_arm_trajectory_generation as trajgen
import flex_arm_trajectory_tracking as trajtrack
import flex_arm_animation as animation
import plot as plot
from simulation_data_handler import save_simulation_data, load_simulation_data



##################################
# SELECTION OF TASKS             #
##################################

task1 = True              # Step trajectory tracking computation activate
task2 = True              # Smooth trajectory computation activate 
task3 = True              # LQR trajectory tracking computation activate also task 2   
task4 = True              # MPC trajectory tracking computation activate also task 2         
task5 = True              # Animation of the double pendulum with reference and optimal trajectory for all tasks



##################################
# INITIALIZE THE VARIABLES       #
##################################

# Number of states and inputs
ns = params.ns
nu = params.nu
TT = params.TT
dt = params.tf/params.TT

# Equilibrium points for task1 and task2 
xx_eq_1 = params.xx_eq_1
xx_eq_2 = params.xx_eq_2
xx_eq_3 = params.xx_eq_3
xx_eq_4 = params.xx_eq_4

# Reference trajectory type for task 1
type_ref_traj_T1 = 'step'  
# Reference trajectory type for task 2
type_ref_traj_T2 = 'cs_w3p'  # Choose between 'step',  'cs', 'cs_w3p', 'cs_w7p'


# Perturbation for LQR
perturbation_LQR = 'medium' # Choose between 'none', 'small', 'medium', 'large'

# Perturbation for MPC
perturbation_MPC = 'medium' # Choose between 'none', 'small', 'medium', 'large'

#Simultìation parameter 
stepsize_0 = trajgen.stepsize_0
maxiters = params.maxiters
cc = trajgen.cc
beta = trajgen.beta
armijo_maxiters = trajgen.armijo_maxiters


flag_plot_ref = params.flag_plot_ref
flag_plot_descent = params.flag_plot_descent
flag_error_ref = params.flag_error_ref
flag_anim = params.flag_anim
flag_plot_vel = params.flag_plot_vel
flag_MPC_difference_constr = params.flag_MPC_difference_constr

# The simulation_review flag is used to check if the simulation data is already saved.
# If the data is already saved, it will load the data instead of running the simulation again.
simulation_review = params.simulation_review   # set to True to load the simulation data

print(colored("\n ------------OPTIMAL CONTROL OF A DOUBLE PENDULUM------------\n", "red", attrs=["bold", "underline"]).center(80))

##################################
#         UTILITY FUNZIONI       #
##################################

def compute_equilibria_and_refs(type_ref_traj, xx_eq_1, xx_eq_2):
    print(colored("\n ------------COMPUTING EQUILIBRIA------------\n", "blue", attrs=["bold"]))
    
    print(colored("\n ------------1st Equilibrium----- \n ", "blue", attrs=["bold", "underline"]))
    u_eq_1 = compeq.compute_equilibria(xx_eq_1)
    print("State Equilibrium: x1 = ", xx_eq_1[0], "x2 = ", xx_eq_1[1], "x3 = ", xx_eq_1[2], "x4 = ", xx_eq_1[3] )
    print("Input Equilibrium: u = ", u_eq_1)

    print(colored("\n ------------2nd Equilibrium----- \n ", "blue", attrs=["bold", "underline"]))
    u_eq_2 = compeq.compute_equilibria(xx_eq_2)
    print("State Equilibrium: x1 = ", xx_eq_2[0], "x2 = ", xx_eq_2[1], "x3 = ", xx_eq_2[2], "x4 = ", xx_eq_2[3] )
    print("Input Equilibrium: u = ", u_eq_2)

    print(colored(f"\nInitializing reference trajectory: {type_ref_traj}", "blue"))
    print(colored("\n ------------REFERENCE TRAJECTORY------------\n", "blue", attrs=["bold", "underline"]))
    print(colored(f"\n ------------Initializing the {type_ref_traj} trajectory------------\n", "blue", attrs=["bold", "underline"]))
    print(colored(f"\n ------------Generate the reference trajectory------------\n", "blue", attrs=["bold", "underline"]))
    xx_ref, uu_ref, type_ref_traj, _, t = trajgen.reference_curve(xx_eq_1, u_eq_1, xx_eq_2, u_eq_2, type_ref_traj)

    if flag_plot_ref:
        
        plot.plot_reference_curve(xx_ref, uu_ref, type_ref_traj, ns, t)
    
    return xx_ref, uu_ref, u_eq_1, u_eq_2, t

def New_trajectory_optimization(xx_ref, uu_ref):
    print(colored("\n ------------RUNNING NEWTON TRAJECTORY OPTIMIZATION------------\n", "blue", attrs=["bold"]))
    xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk = trajgen.traj_gen_newton(xx_ref, uu_ref)
    return xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk

def LQR_trajectory_optimization(xx_ref, uu_ref):
    print(colored("\n ------------RUNNING LQR TRAJECTORY OPTIMIZATION------------\n", "blue", attrs=["bold"]))
    xx_opt_lqr, uu_opt_lqr, JJ_lqr, delta_x_norm  = trajtrack.LQR_trajectory_tracking(xx_ref, uu_ref)
    return xx_opt_lqr, uu_opt_lqr, JJ_lqr, delta_x_norm 

def MPC_trajectory_optimization(xx_opt_smooth, uu_opt_smooth, perturbation_MPC):
    print(colored("\n ------------RUNNING MPC TRAJECTORY OPTIMIZATION------------\n", "blue", attrs=["bold"]))
    xx_opt_MPC, uu_opt_MPC, JJ_MPC  = trajtrack.MPC_trajectory_tracking(xx_opt_smooth, uu_opt_smooth, perturbation_MPC)
    return xx_opt_MPC, uu_opt_MPC, JJ_MPC 

def plot_results(xx_ref, uu_ref, xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk, t, title, cost = True):

    print(colored("\n ------------Plot the reference and optimal trajectories------------ \n", "blue", attrs=["bold", "underline"]))
    
    #plot.plot_comparison(xx_opt, xx_ref, TT, title)
    if ((task1 or task2) and not (task3 or task4 or simulation_review)):
        plot.plot_comparison(xx_plot, xx_ref, TT, title) #plotting iterations of first states
    elif (task3 or task4 or simulation_review):
        plot.plot_comparison(xx_opt, xx_ref, TT, title) #plotting first state

    
    print(colored("\n ------------Plot the reference and optimal trajectories for each states------------ \n", "blue", attrs=["bold", "underline"]))
    #evolution of first 2 states: position
    if flag_plot_ref:
        plot.plot_trajectory(xx_opt, xx_ref, TT, title)
    
    error = np.linalg.norm(xx_ref - xx_opt, axis=0)
    mean_error = np.mean(error)

    print(f"------------Mean error: {mean_error:.4f}------------\n")

    if flag_error_ref:
        print(colored("\n ------------PLOT ERROR BETWEEN REFERENCE AND OPTIMAL TRAJECTORY----------\n ", "blue", attrs=["bold", "underline"]))
        plot.error_ref(error, title)


    # Plot Cost Value and 
    if cost == True and not simulation_review:
        print(colored("\n ------------Plot Cost Value------------ \n", "blue", attrs=["bold", "underline"]))
        plot.plot_cost_function(JJ, title)
        print(colored(f"[Iteration {kk}] Cost: {JJ[kk]} | Norm of descent direction: {descent[kk]}", "blue", attrs=["bold"]))

    if flag_anim and task5:
        print(colored("\n ------------Animation of the double pendulum with reference and optimal trajectory------------\n", "blue", attrs=["bold", "underline"]))
        animation.animate_double_pendulum(xx_opt, xx_ref,title, dt=1e-3)

def load_or_run_task(path, type_ref_traj):
    
    if simulation_review and os.path.exists(path):
        data = load_simulation_data(path)
        print(colored("Simulation data loaded successfully!", "green", attrs=["bold", "underline"]))

        # Extract loaded data
        parameters = data.get('parameters', {})
        if parameters:
            print(colored("Simulation Parameters: \n ", "blue"))
            for key, value in parameters.items():
                print(f"{key}: {value}")

        return data['xx_ref'], data['uu_ref'], data['u_eq_1'], data['u_eq_2'], data['xx_opt'], data['uu_opt'], data['xx_plot'], \
               data['uu_plot'], data['JJ'], data['descent'], data['kk'], data['t']
    else:
        # If no saved simulation data is found, run the simulation
        print(colored("No saved simulation data found, running the simulation... \n ", "yellow", attrs=["bold"]))
        xx_ref, uu_ref, u_eq_1, u_eq_2, t = compute_equilibria_and_refs(type_ref_traj, xx_eq_1, xx_eq_2)
        xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk = New_trajectory_optimization(xx_ref, uu_ref)

        save_simulation_data(path, {
            'xx_ref': xx_ref, 'uu_ref': uu_ref, 'u_eq_1': u_eq_1, 'u_eq_2': u_eq_2,'xx_opt': xx_opt, 'uu_opt': uu_opt,
            'xx_plot': xx_plot, 'uu_plot': uu_plot, 'JJ': JJ, 'descent': descent, 'kk': kk, 't': t
        })
        # Save the simulation data
        print(colored("\n Simulation data saved successfully!", "green", attrs=["bold", "underline"]))
        return xx_ref, uu_ref, u_eq_1, u_eq_2, xx_opt, uu_opt, xx_plot, uu_plot, JJ, descent, kk, t 

def main():

    global simulation_review, perturbation_LQR, perturbation_MPC

    ###########################################
    # TASK 1 - STEP TRAJECTORY TRACKING       #
    ###########################################
    if task1:
        
        title = "TASK 1"
        # Path to save the simulation data
        simulation_data_file = 'code\Simulation_Data\Task_1\simulation_step_10.plk'
        print(colored("\n ------------STARTING TASK 1 - STEP TRAJECTORY TRACKING------------\n", "blue", attrs=["bold", "underline"]).center(80))

        if simulation_review == True:
            # Load the simulation data
            xx_ref, uu_ref, u_eq_1, u_eq_2, xx_opt, uu_opt, xx_plot, uu_plot, JJ, descent, kk, t = load_or_run_task(simulation_data_file, type_ref_traj_T1)
        else:
            # Compute the equilibrium points and reference trajectory
            xx_ref, uu_ref, u_eq_1, u_eq_2, t = compute_equilibria_and_refs(type_ref_traj_T1, xx_eq_1, xx_eq_2)
            #Generate optimal trajectory
            xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk = New_trajectory_optimization(xx_ref, uu_ref)
        # Plot results
        plot_results(xx_ref, uu_ref, xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk, t, title, cost=True)



    ###########################################q
    # TASK 2 - SMOOTH TRAJECTORY TRACKING     #
    ###########################################
    
    if task2:
        title = "TASK 2"

        # Path to save the simulation data for Task 2
        simulation_data_task2 = 'code\Simulation_Data\Task_2\simulation_cs_w3p_10.plk'

        print(colored("\n ------------STARTING TASK 2 - SMOOTH TRAJECTORY TRACKING------------\n", "blue", attrs=["bold"]))

        if simulation_review:
            # xx_ref, uu_ref, u_eq_1, u_eq_2, xx_opt, uu_opt, xx_plot, uu_plot, JJ, descent, kk, t 
            xx_ref_smooth, uu_ref_smooth, u_eq_1, u_eq_2, xx_opt_smooth, uu_opt_smooth,JJ_smooth, xx_plot_smooth, uu_plot_smooth, descent_smooth, kk_smooth, t = load_or_run_task(simulation_data_task2, type_ref_traj_T2)
            
        else:
            # Compute the equilibrium points and reference trajectory
            xx_ref_smooth, uu_ref_smooth, u_eq_1, u_eq_2, t = compute_equilibria_and_refs(type_ref_traj_T2, xx_eq_1, xx_eq_2)
            # Generate smooth trajectory
            print(colored("\n ------------Generate smooth trajectory using Newton's method------------\n", "blue", attrs=["bold"]))
            # Generate the smooth trajectory using Newton's method
            xx_opt_smooth, uu_opt_smooth,JJ_smooth,  xx_plot_smooth, uu_plot_smooth, descent_smooth, kk_smooth = New_trajectory_optimization(xx_ref_smooth, uu_ref_smooth)  
        # Plot results
        plot_results(xx_ref_smooth, uu_ref_smooth, xx_opt_smooth, uu_opt_smooth, JJ_smooth, xx_plot_smooth, uu_plot_smooth, descent_smooth, kk_smooth, t, title, cost=True)




    ###########################################
    # TASK 3 - LQR TRAJECTORY TRACKING        #
    ###########################################

    if  task3:
        title = "TASK 3"
        
        print(colored("\n ------------STARTING TASK 3 - LQR TRAJECTORY TRACKING------------\n", "blue", attrs=["bold"]))

        # Load the simulation data from task 2
        if simulation_review:
            xx_ref_smooth, uu_ref_smooth, u_eq_1, u_eq_2,xx_opt_smooth, uu_opt_smooth, xx_plot_smooth, uu_plot_smooth, JJ_smooth, descent_smooth, kk_smooth, t = load_or_run_task(simulation_data_task2, type_ref_traj_T2)

        if perturbation_LQR == 'none':
            perturbation_LQR = np.array([0.0, 0.0, 0.0, 0.0])

        elif perturbation_LQR == 'small':
            perturbation_LQR = params.perturbation_LQR_small

        elif perturbation_LQR == 'medium':
            perturbation_LQR = params.perturbation_LQR_medium

        elif perturbation_LQR == 'large':
            perturbation_LQR = params.perturbation_LQR_large

        print(colored("\n ------------Generate smooth trajectory using LQR_trajectory_tracking------------\n", "blue", attrs=["bold"]))
        xx_opt_lqr, uu_opt_lqr, JJ_lqr, delta_x_norm = trajtrack.LQR_trajectory_tracking(xx_opt_smooth, uu_opt_smooth, perturbation_LQR, extra=True)

        plot_results(xx_opt_smooth, uu_opt_smooth, xx_opt_lqr, uu_opt_lqr, JJ_lqr, xx_plot_smooth, uu_plot_smooth, delta_x_norm, kk_smooth, t, title, cost=False)



    ###########################################
    # TASK 4 - MPC TRAJECTORY TRACKING        #
    ###########################################

    if task4:
        title = "TASK 4"

        if simulation_review:
            xx_ref_smooth, uu_ref_smooth, u_eq_1, u_eq_2, xx_opt_smooth, uu_opt_smooth, xx_plot_smooth, uu_plot_smooth, JJ_smooth, descent_smooth, kk_smooth, t = load_or_run_task(simulation_data_task2, type_ref_traj_T2)
        elif not simulation_review and not task2:
            print(colored("\n ------------Please run Task 2 before Task 4 or activate the flag for loading simulation data------------\n", "red", attrs=["bold"]))
            sys.exit(1)
            

        print(colored("\n ------------STARTING TASK 4 - MPC TRAJECTORY TRACKING------------\n", "blue", attrs=["bold"]))

        if perturbation_MPC == 'none':
            perturbation_MPC = np.array([0.0, 0.0, 0.0, 0.0])

        elif perturbation_MPC == 'small':
            perturbation_MPC = params.perturbation_MPC_small

        elif perturbation_MPC == 'medium':
            perturbation_MPC = params.perturbation_MPC_medium

        elif perturbation_MPC == 'large':
            perturbation_MPC = params.perturbation_MPC_large

        extra = True #extra perturbation in MPC

        print(colored("\n ------------Generate smooth trajectory using MPC_trajectory_tracking------------\n", "blue", attrs=["bold"]))
        
        if flag_MPC_difference_constr == False: #WITH constraints on input
             #Enabled constraints on input
            xx_opt_mpc, uu_opt_mpc, JJ_mpc = trajtrack.MPC_trajectory_tracking(xx_opt_smooth, uu_opt_smooth, perturbation_MPC, extra, flag_constr_input_MPC = True)
        elif flag_MPC_difference_constr == True:
            # ENabled constraints on input
            xx_opt_mpc, uu_opt_mpc, JJ_mpc = trajtrack.MPC_trajectory_tracking(xx_opt_smooth, uu_opt_smooth, perturbation_MPC, extra, flag_constr_input_MPC = True)
            # Disabled
            xx_opt_mpc_NOConstr, uu_opt_mpc_NOConstr, JJ_mpc_NOConstr = trajtrack.MPC_trajectory_tracking(xx_opt_smooth, uu_opt_smooth, perturbation_MPC, extra, flag_constr_input_MPC=False)
       

        # Plot results
        plot_results(xx_opt_smooth, uu_opt_smooth, xx_opt_mpc, uu_opt_mpc, JJ_mpc, xx_plot_smooth, uu_plot_smooth, descent_smooth, kk_smooth, t, title, cost=False)

        if flag_MPC_difference_constr:
            print(colored("\n ------------Plot difference between MPC With VS Without contraints on input------------ \n", "blue", attrs=["bold", "underline"]))
            title = 'MPC State x[0] with VS without input contraints'
            plot.plot_comparison(xx_opt_mpc, xx_opt_mpc_NOConstr, TT, title)
            title = 'MPC INPUT with VS without input contraints'
            plot.plot_comparison_MPC(uu_opt_mpc, uu_opt_mpc_NOConstr, TT, title)

    ###########################################
    # Comparison LQR e MPC                    #
    ###########################################

    if task3 and task4:
        print(colored("\n ------------COMPARISON BETWEEN LQR AND MPC TRAJECTORY TRACKING------------\n", "blue", attrs=["bold"]))
        title = "LQR vs MPC"
        # Plot comparison between LQR and MPC - state: theta1
        plot.plot_comparison_LQR_MPC_1(xx_opt_smooth, xx_opt_lqr, xx_opt_mpc, TT, title)
        # Plot comparison between LQR and MPC - states theta1 and theta2 
        plot.plot_comparison_LQR_MPC_2(xx_opt_smooth, xx_opt_lqr, xx_opt_mpc, TT, title)
        # Error for LQR 
        error_lqr = np.linalg.norm(xx_opt_smooth - xx_opt_lqr, axis=0)
        mean_error_lqr = np.mean(error_lqr)
        print(f"------------Mean error LQR: {mean_error_lqr:.4f}------------\n")
        # Error for MPC
        error_mpc = np.linalg.norm(xx_opt_smooth - xx_opt_mpc, axis=0)
        mean_error_mpc = np.mean(error_mpc)
        print(f"------------Mean error MPC: {mean_error_mpc:.4f}------------\n")


if __name__ == "__main__":
    main()

    print(colored("\n ------------END OF SIMULATION------------\n", "green", attrs=["bold", "underline"]).center(80))
    print(colored("\n ------------THANK YOU FOR RUNNING------------\n", "green", attrs=["bold", "underline"]).center(80))