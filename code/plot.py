import numpy as np
import matplotlib.pyplot as plt

import flex_arm_parameters as params
import flex_arm_dynamics as dyn
import flex_arm_cost as cst


flag_plot_vel = params.flag_plot_vel



#Plot the reference curve for a pendulum based on the given equilibrium points and type of reference trajectory.
def plot_reference_curve(xx_ref, uu_ref, type_ref_traj, ns, t):
    """
    Plot the reference curve for a pendulum based on the given equilibrium points 
    and type of reference trajectory.
    
    Parameters:
        - xx_ref: Generated state reference trajectory.
        - uu_ref: Generated input reference trajectory.
        - type_ref_traj: Type of trajectory (e.g. "step", "cs", "poly5", "loop" ).
        - ns: Number of states.
        - t: Time vector for the reference trajectory.

    Returns:
        None: Displays the plots of the reference trajectory and state variables.
    """

    flag_plot_ref = params.flag_plot_ref
    state_variable = params.state_variable  # plot the state variables
   
    if flag_plot_ref:

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Reference trajectory {type_ref_traj}', fontsize=16)
        axs[0].plot(np.linspace(0, params.tf, params.TT), xx_ref[0, :], 'r--')
        axs[0].set_ylabel('Theta 1 [rad]', fontsize=14)
        plt.grid()
        axs[1].plot(np.linspace(0, params.tf, params.TT), xx_ref[1, :], 'b--')
        axs[1].set_ylabel('Theta 2 [rad]', fontsize=14)
        axs[1].set_xlabel('Time [s]', fontsize=14)
        #plt.legend()
        plt.grid()
        plt.show()

    if state_variable:
        plt.figure(figsize=(10, 8))
        for i in range(ns):
            plt.plot(t, xx_ref[i, :] , label=f'State variable xx{i+1}')
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("State Variables", fontsize=14)
        plt.title(f"{type_ref_traj.capitalize()} Reference Curve of namber of states {ns}", fontsize=16)
        plt.legend()
        plt.grid()

        plt.figure(figsize=(10, 8))
        plt.plot(t, uu_ref[0, :], 'g--',label="Input 1 (uu[0])")
        plt.xlabel("Time [s]", fontsize=14)
        plt.ylabel("Input Variables", fontsize=14)
        plt.title(f"{type_ref_traj.capitalize()} Input Curve", fontsize=16)
        #plt.legend()
        plt.grid()
        plt.show()


def plot_cost_function(cost_values, title):

    """
    Plots the cost function values over iterations in semilog scale.

    Parameters:
        - cost_values (list): List of cost function values.
        - title (str): Title for the plot.

    Returns:
        - None
    """
    if isinstance(cost_values, list) or cost_values.ndim == 1:
        cost_values = np.array(cost_values)
    elif cost_values.ndim == 2:
        cost_values = np.sum(cost_values, axis=0)
    elif cost_values.ndim == 3:
        cost_values = np.sum(cost_values, axis=(0, 1))  

    # print("cost_values.shape: ", cost_values.shape)


    plt.figure()
    plt.semilogy(abs(cost_values), color='k', marker = 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost function')
    plt.title("Cost function over iterations - " + title)
    plt.grid()
    plt.show()

    return None

"""

def descent_direction_plot(descent, max_iters):
    '''
    Plot the descent direction based on the number of iterations and the descent values.
 
    Parameters:
        - descent (list): List of descent direction values.
        - max_iters (int): Maximum number of iterations to plot.
 
    return: 
        - None
    '''

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(max_iters), descent[:max_iters])
    plt.xlabel('Iteration Number', fontsize=14)
    plt.ylabel('Descent Direction', fontsize=14)
    plt.yscale('log')
    plt.title('Descent Direction Plot', fontsize=16)
    plt.grid()
    plt.show()

"""

def semilog_descent_plot(descent):

    """
    Plots the norm of the descent direction in a semilog scale against the iterations.

    Parameters:
        - descent (list): Norm of descent direction at each iteration.

    Returns:
        - None
    """


    # Remove elements which have not been filled (still initialized to zero)
    descent = np.array([val for val in descent if val != 0])

    plt.figure()
    plt.semilogy(abs(descent), color='k', marker = 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Descent direction')
    plt.title("Descent direction over iterations")
    plt.grid()
    plt.show()  

# Plot the trajectory of the system over time. for sigle state variables
def plot_trajectory(xx_opt, xx_ref, TT, title):
    """
    Plot the trajectory of the system over time for single state variables.
    Parameters:
        - xx_opt: Optimal trajectory data (shape: states x TT).
        - xx_ref: Reference trajectory data (shape: states x TT).
        - TT: Number of time samples.
        - title: Title for the plot.

    Returns:
        - None
    """
    t = np.arange(TT)  # Timeline

    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=16)
    # Plot per θ1
    plt.subplot(1, 2, 1)
    plt.plot(t, xx_opt[0, :], label="θ1 (Optimal)", linestyle='-', color='b')
    plt.plot(t, xx_ref[0, :], label="θ1 (Reference)", linestyle='--', color='r')
    plt.xlabel("Time")
    plt.ylabel("Angle θ1 (rad)")
    plt.title("Evolution of θ1")
    plt.legend()
    plt.grid(True)

    # Plot per θ2
    plt.subplot(1, 2, 2)
    plt.plot(t, xx_opt[1, :], label="θ2 (Optimal)", linestyle='-', color='b')
    plt.plot(t, xx_ref[1, :], label="θ2 (Reference)", linestyle='--', color='r')
    plt.xlabel("Time")
    plt.ylabel("Angle θ2 (rad)")
    plt.title("Evolution of θ2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    if flag_plot_vel:
        plt.figure(figsize=(12, 5))
        plt.suptitle(title, fontsize=16)
        # Plot per θ1 dot
        plt.subplot(1, 2, 1)
        plt.plot(t, xx_opt[2, :], label="dθ1/dt (Optimal)", linestyle='-', color='b')
        plt.plot(t, xx_ref[2, :], label="dθ1/dt (Reference)", linestyle='--', color='r')
        plt.xlabel("Time")
        plt.ylabel("Velocity dθ1/dt (rad/s)")
        plt.title("Evolution of dθ1/dt")
        plt.legend()
        plt.grid(True)

        # Plot per θ2 dot
        plt.subplot(1, 2, 2)
        plt.plot(t, xx_opt[3, :], label="dθ2/dt (Optimal)", linestyle='-', color='b')
        plt.plot(t, xx_ref[3, :], label="dθ2/dt (Reference)", linestyle='--', color='r')
        plt.xlabel("Time")
        plt.ylabel("Velocity dθ2/dt (rad/s)")
        plt.title("Evolution of dθ2/dt")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def plot_comparison(xx_plot, xx_ref, TT, title):
    """
    Plot the comparison between the optimal trajectory and the reference trajectory.
 
    Parameters:
        - xx_plot: Optimal trajectory data.
        - xx_ref: Reference trajectory data.
        - TT: Number of time samples.
        - title: Title for the plot.
 
    Returns:
        - None
    """
    if params.print_debug:
        print("-------------xx_plot.shape:\n", xx_plot.shape)
        print("-------------xx_ref.shape:\n", xx_ref.shape)
        print("--------------TT:\n ", TT)

    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Optimal VS Reference trajectory - '+ title, fontsize=16)
 
    # Plot x[0] and x_ref[0] on the first subplot (no subplots)
    num_trajectories = xx_plot.shape[2] if xx_plot.ndim == 3 else 1
    for k in range(num_trajectories):
        if num_trajectories == 1:
            plt.plot(np.linspace(0, TT-1, TT), xx_plot[0, :], label="x[0]", color='b')  # Optimal trajectory
        else:
            plt.plot(np.linspace(0, TT-1, TT), xx_plot[0, :, k], label="x[0]^(k="+str(k)+")")  # Optimal trajectory
    plt.plot(np.linspace(0, TT-1, TT), xx_ref[0, :], "--", label="$x_{ref}[0]$", color='r')  # Reference trajectory
    plt.grid()
    plt.legend()
    plt.title("Comparison of x[0] and $x_{ref}[0]$")
    plt.xlabel("Time steps")
    plt.ylabel("x[0]")
   
    # Show the plot
    plt.show()

def plot_comparison_MPC(uu_plot, uu_ref, TT, title):
    """
    Plot the comparison between the optimal trajectory and the reference trajectory.
 
    Parameters:
        - uu_plot: Optimal trajectory data.
        - uu_ref: Reference trajectory data.
        - TT: Number of time samples.
        - title: Title for the plot.
 
    Returns:
        - None
    """
    uu_min = params.u1_min
    uu_max = params.u1_max
    if params.print_debug:
        print("-------------uu_plot.shape:\n", uu_plot.shape)
        print("-------------uu_ref.shape:\n", uu_ref.shape)
        print("--------------TT:\n ", TT)

    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Input Constraints VS NO input Constraints - '+title, fontsize=16)
 
    # Plot x[0] and x_ref[0] on the first subplot (no subplots)
    num_trajectories = uu_plot.shape[2] if uu_plot.ndim == 3 else 1
    if num_trajectories == 1:
        plt.plot(np.linspace(0, TT-1, TT), uu_plot[0, :], label="u with", color='b')  # Optimal trajectory

    plt.plot(np.linspace(0, TT-1, TT), uu_ref[0, :], "--", label="$u without$", color='r')
    y = np.ones_like(np.linspace(0, TT-1, TT)) * uu_min
    plt.plot(np.linspace(0, TT-1, TT), y, label=f'u_min ', color = 'green')
    y = np.ones_like(np.linspace(0, TT-1, TT)) * uu_max
    plt.plot(np.linspace(0, TT-1, TT), y, label=f'u_max ', color = 'green')
    plt.grid()
    plt.legend()
    plt.title("Comparison of u with Constraints and $u without$")
    plt.xlabel("Time steps")
    plt.ylabel("u")
   
    # Show the plot
    plt.show()
    


def optimal_trajectory(xx_plot,xx_ref,TT): 
    """
    Plot the optimal trajectory and the reference trajectory for the first state variable.
    Parameters:
        - xx_plot: Optimal trajectory data (shape: states x TT).
        - xx_ref: Reference trajectory data (shape: states x TT).
        - TT: Number of time samples.
    Returns:
        - None
    """
    # Plot x[0] and x_ref[0] on the first subplot
    plt.figure()
    plt.plot(np.linspace(0, TT-1, TT), xx_plot[0,:], "b-", label="x[0]")
    plt.plot(np.linspace(0, TT-1, TT), xx_ref[0, :], "b--", label="$x_{ref}[0]$")
    plt.grid()
    plt.legend()
    plt.title("Comparison of x[0] and $x_{ref}[0]$")
    plt.show()


def error_ref(error, title):

    """Plot the error between the reference trajectory and the optimal trajectory.
    Parameters:
        - error: Error data (shape: TT).
        - title: Title for the plot.
    Returns:
        - None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(error, label='Error over time')
    plt.xlabel('Time step')
    plt.ylabel('Error')
    plt.title('Difference between Reference and Optimal Trajectories - ' + title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparison_LQR_MPC_1(xx_ref,xx_LQR, xx_MPC, TT, title):
    """
    Plot the comparison between the LQR, MPC, and reference trajectories.

    Parameters:
        - xx_LQR: Trajectory data from LQR controller (shape: states x TT).
        - xx_MPC: Trajectory data from MPC controller (shape: states x TT).
        - xx_ref: Reference trajectory (shape: states x TT).
        - TT: Number of time samples.
        - title: Title for the plot.

    Returns:
        - None
    """
    t = np.linspace(0, TT-1, TT)
    plt.figure(figsize=(12, 5))
    plt.suptitle('LQR vs MPC vs Reference Trajectory Comparison - ' + title, fontsize=16)

    # Plot for state 1
    plt.plot(t, xx_LQR[0, :], label="LQR x[0]", color='b')
    plt.plot(t, xx_MPC[0, :], label="MPC x[0]", color='g')
    plt.plot(t, xx_ref[0, :], label="Reference x[0]", color='r', linestyle='--')
    plt.xlabel("Time steps")
    plt.ylabel("x[0]")
    plt.title("State x[0]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_comparison_LQR_MPC_2(xx_ref,xx_LQR, xx_MPC, TT, title):
    """
    Plot the comparison between the LQR, MPC, and reference trajectories.

    Parameters:
        - xx_LQR: Trajectory data from LQR controller (shape: states x TT).
        - xx_MPC: Trajectory data from MPC controller (shape: states x TT).
        - xx_ref: Reference trajectory (shape: states x TT).
        - TT: Number of time samples.
        - title: Title for the plot.

    Returns:
        - None
    """
    t = np.linspace(0, TT-1, TT)
    plt.figure(figsize=(12, 5))
    plt.suptitle('LQR vs MPC vs Reference Trajectory Comparison - ' + title, fontsize=16)

    # Plot for state 1
    plt.subplot(1, 2, 1)
    plt.plot(t, xx_LQR[0, :], label="LQR x[0]", color='b')
    plt.plot(t, xx_MPC[0, :], label="MPC x[0]", color='g')
    plt.plot(t, xx_ref[0, :], label="Reference x[0]", color='r', linestyle='--')
    plt.xlabel("Time steps")
    plt.ylabel("x[0]")
    plt.title("State x[0]")
    plt.legend()
    plt.grid(True)

    # Plot for state 2 (if available)
    if xx_LQR.shape[0] > 1 and xx_MPC.shape[0] > 1 and xx_ref.shape[0] > 1:
        plt.subplot(1, 2, 2)
        plt.plot(t, xx_LQR[1, :], label="LQR x[1]", color='b')
        plt.plot(t, xx_MPC[1, :], label="MPC x[1]", color='g')
        plt.plot(t, xx_ref[1, :], label="Reference x[1]", color='r', linestyle='--')
        plt.xlabel("Time steps")
        plt.ylabel("x[1]")
        plt.title("State x[1]")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    