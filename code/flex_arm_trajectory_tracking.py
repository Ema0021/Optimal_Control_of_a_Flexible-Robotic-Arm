import numpy as np
from termcolor import colored
import cvxpy as cp # For Model Predictive Control (MPC) library to solve the optimization problem 

import flex_arm_dynamics as dyn
import flex_arm_parameters as params
import flex_arm_cost as cost

# Import parameters
ns = params.ns
nu = params.nu
TT = params.TT

print_debug = params.print_debug


QQt_lqr = cost.QQt_lqr
RRt_lqr = cost.RRt_lqr
QQT_lqr = cost.QQT_lqr

QQt_mpc = cost.QQt_mpc
RRt_mpc = cost.RRt_mpc
QQT_mpc = cost.QQT_mpc




def LQR_trajectory_tracking(xx_opt, uu_opt, perturbation_LQR, extra=False):
    """
    Performs trajectory tracking using LQR control.

    Parameters:
        - xx_opt (numpy.ndarray): State trajectory to track.
        - uu_opt (numpy.ndarray): Control input trajectory to track.
        - perturbation_LQR (numpy.ndarray): Perturbation to be added to the initial state.
        - extra (bool): If True, adds an extra perturbation at the middle of the simulation.

    Returns:
        - xx (numpy.ndarray): State trajectory tracking the optimal trajectory.
        - uu (numpy.ndarray): Control input trajectory tracking the optimal trajectory.
        - JJ_lqr_list (list): List of cost function values at each time step.
        - delta_u_list (list): List of control input perturbations at each time step.
    """
    
    # Import parameters
    QQ = QQt_lqr
    RR = RRt_lqr
    QQT = QQT_lqr


    # Initialize matrices
    AA = np.zeros((ns, ns, TT))
    BB = np.zeros((ns, nu, TT))

    delta_x_list = []
    delta_u_list = []
    
    # 1. Linearize the dynamics about the optimal trajectory
    for tt in range(TT-1):

        fx, fu = dyn.dynamics(xx_opt[:,tt], uu_opt[:,tt])[1:]  # Linearization
        AA[:,:,tt] = fx.T
        BB[:,:,tt] = fu.T

    # 2. Solve the LQR problem 
    KK = np.zeros((nu, ns, TT))
    PP = np.zeros((ns, ns, TT))

    PP[:,:,-1] = QQT

    for tt in reversed(range(TT-1)):
        QQt = QQ
        RRt = RR
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]

        # Compute the feedback gain K[t]
        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        KK[:,:,tt] = -MMt_inv @ (BBt.T @ PPtp @ AAt)

        # Update Riccati matrix
        PP[:,:,tt] = QQt + AAt.T @ PPtp @ AAt - KK[:,:,tt].T @ (RRt + BBt.T @ PPtp @ BBt) @ KK[:,:,tt]

    # 3. Track the optimal trajectory
    xx = np.tile(xx_opt[:,0], (TT, 1)).T
    uu = np.tile(uu_opt[:,0], (TT, 1)).T

    # Introduce a perturbation to show tracking properties of the LQR
    xx[:,0] = xx_opt[:,0].squeeze() + perturbation_LQR

    JJ = 0 # Cost function value
    JJ_lqr_list = []
    

    for tt in range(TT-1):
        # Calculate the control law: u = u_ref + K*(x - x_ref)
        delta_x = xx[:, tt] - xx_opt[:, tt]
        uu[:, tt] = uu_opt[:, tt] + KK[:,:,tt] @ delta_x
        delta_u = uu[:, tt] - uu_opt[:, tt]
        
        delta_u_norm = delta_u.T @ delta_u
        delta_x_list.append(delta_x)
        delta_u_list.append(delta_u_norm)

        stete_error = xx[:, tt] - xx_opt[:, tt]
        control_error = uu[:, tt] - uu_opt[:, tt]

        cst = 0.5*(stete_error.T @ QQ @ stete_error + control_error.T @ RR @ control_error)
        JJ += cst 
        JJ_lqr_list.append(cst)

        if extra and tt == TT // 2:
            print(colored(f"Extra perturbation at time step {tt} with value {perturbation_LQR}", 'yellow', attrs=['bold']))
            xx[:, tt] += perturbation_LQR
        xx[:, tt+1] = dyn.dynamics(xx[:, tt], uu[:, tt])[0].flatten()  # Propagate the state
    

    JJ += 0.5*(xx[:,-1] - xx_opt[:,-1]).T @ QQT @ (xx[:,-1] - xx_opt[:,-1])  # Terminal cost
    JJ_lqr_list.append(JJ)

    # Print the total cost at the end
    print(colored(f"\n Total cost for the tracking trajectory with LQR : {JJ.item()} \n ", 'blue', attrs=['bold']))
    # For plot purposes, keep the last control input and state constant
    uu[:,-1] = uu[:,-2]
    xx[:,-1] = xx[:,-2]
   
    return xx, uu, JJ_lqr_list, delta_u_list



def MPC_solve_step(AA, BB, QQ, RR, QQT, xx0, xx_ref,uu_ref, x1_min, x1_max, x2_min, x2_max, u1_min, u1_max, flag_constr_input_MPC, T_pred):
    """
    Solve trajectory tracking using Model Predictive Control (MPC). Inner loop of the MPC algorithm.
    
    Parameters:
        - AA (numpy.ndarray): State matrix.
        - BB (numpy.ndarray): Input matrix.
        - QQ (numpy.ndarray): State cost matrix.
        - RR (numpy.ndarray): Input cost matrix.
        - QQT (numpy.ndarray): Terminal state cost matrix.
        - xx0 (numpy.ndarray): Initial state.
        - xx_ref (numpy.ndarray): Reference state trajectory.
        - uu_ref (numpy.ndarray): Reference control input trajectory.
        - x1_min (float): Minimum value for state variable x1 -> theta1.
        - x1_max (float): Maximum value for state variable x1 -> theta1.
        - x2_min (float): Minimum value for state variable x2 -> theta2.
        - x2_max (float): Maximum value for state variable x2 -> theta2.
        - u1_min (float): Minimum value for control input u1 -> torque1.
        - u1_max (float): Maximum value for control input u1 -> torque1.
        - flag_constr_input_MPC (bool): Flag to indicate whether to apply constraints on the input.
        - T_pred (int): Prediction horizon.

    
    Returns:
        - uu_mpc (numpy.ndarray): Control input trajectory.
        - problem.value (float): Cost function value.
        - counter (int): Counter for infeasible problems.
    """

    xx0 = xx0.squeeze()
    samples = AA.shape[2] - 1 

    if params.print_debug: 
        print(f"\n -----------------------SAMPLE: {samples},")

    ns = params.ns
    nu = params.nu

    # 1. Define the optimization variables
    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((nu, T_pred))

    # 2. Initialize cost and constraints
    cst = 0
    constr = []

    # 3. Fill cost and constraint list
    for tau in range(min(T_pred-1, samples)):

        cst += cp.quad_form(xx_mpc[:,tau] - xx_ref[:,tau], QQ) + cp.quad_form(uu_mpc[:,tau] - uu_ref[:,tau], RR)

        if print_debug:
            print("\n TAU:", tau)
            print("SAMPLES:", samples)
            print("TPRED:", T_pred-1)
            print("MIN:", min(T_pred-1, samples))
            print(f"tau -1 : {tau-1}----------------------\n")
            print(f"sasmples -1 : {samples-1}----------------\n")
            print(f"tau+1 : {tau+1}---------------------\n")
            print(f"MINIMO tau+1 PRIMA =  : {min(tau+1, samples-1)}----------------\n")
            print(f"MINIMO tau-1 DOPO = :{min(tau,samples-1)}----------------\n")

        if flag_constr_input_MPC:   #WITH Constraint on input
            constr += [
                    xx_mpc[:,tau+1] - xx_ref[:,min(tau+1, samples)] == AA[:,:,tau]@(xx_mpc[:,tau]- xx_ref[:,min(tau, samples-1)]) + BB[:,:,tau]@(uu_mpc[:,tau] - uu_ref[:,min(tau, samples-1)]),
                    xx_mpc[0,tau] <= x1_max,
                    xx_mpc[0,tau] >= x1_min,
                    xx_mpc[1,tau] <= x2_max,
                    xx_mpc[1,tau] >= x2_min,
                    uu_mpc[0,tau] <= u1_max,
                    uu_mpc[0,tau] >= u1_min,
                    ]
        else :      #NO Constraint on input
            constr += [
    
                    xx_mpc[:,tau+1] - xx_ref[:,min(tau+1, samples)] == AA[:,:,tau]@(xx_mpc[:,tau]- xx_ref[:,min(tau, samples-1)]) + BB[:,:,tau]@(uu_mpc[:,tau] - uu_ref[:,min(tau, samples-1)]),
                    xx_mpc[0,tau] <= x1_max,
                    xx_mpc[0,tau] >= x1_min,
                    xx_mpc[1,tau] <= x2_max,
                    xx_mpc[1,tau] >= x2_min
                    ]
            
        
    constr += [xx_mpc[:,0] == xx0]  # Initial condition
    
    cst += cp.quad_form(xx_mpc[:,T_pred-1] - xx_ref[:,min(T_pred-1, samples-1)], QQT)

    # Hard Constraints
    # constr += [xx_mpc[:,T_pred-1]- xx_ref[:,min(T_pred - 1, samples - 1)] == np.array([0.0,0.0,0.0,0.0])]  # Terminal condition


    # 4. Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cst), constr)
    # Try to solve the problem with different solvers
    # problem.solve()
    # problem.solve(solver=cp.SCS)
    problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)

    counter = 0 # Counter for infeasible problems

    if problem.status == "infeasible":
        print(colored("Infeasible problem! CHECK YOUR CONSTRAINTS!!!", 'red', attrs=['bold']))
        counter += 1
        
    if problem.status != "optimal":
        print(colored(f"MPC infeasible or not solved properly: {problem.status}", 'red', attrs=['bold']))
        

    if print_debug:
        print(f"\nNumber of worning constraints: {len(constr)}\n")

    return uu_mpc[:,0].value, problem.value, counter

def MPC_trajectory_tracking(xx_opt, uu_opt, perturbation_MPC, extra, flag_constr_input_MPC):

    """
    Executes the outer loop of the Model Predictive Control (MPC) algorithm.
    
    Parameters:
        - xx_opt (numpy.ndarray): Optimal state trajectory.
        - uu_opt (numpy.ndarray): Optimal control input trajectory.
        - perturbation_MPC (numpy.ndarray): Perturbation to be added to the initial state.
        - extra (bool): Flag to indicate whether to add an extra perturbation at the middle of the simulation.
        - flag_constr_input_MPC (bool): Flag to indicate whether to apply constraints on the input.
    
    Returns:
        - xx_real (numpy.ndarray): Real state trajectory.
        - uu_real (numpy.ndarray): Real control input trajectory.
        - JJ_mpc_list (list): List of cost function values at each time step.
    
    """

    # Import parameters
    T_pred_mpc = params.T_pred_mpc   # Time for the internal cycle of the MPC
    T_sim = params.T_sim_mpc         # Time for the overall simulation of the MPC
    x1_min = params.x1_min
    x1_max = params.x1_max
    x2_min = params.x2_min
    x2_max = params.x2_max
    u1_min = params.u1_min
    u1_max = params.u1_max


    QQt_mpc = cost.QQt_mpc
    RRt_mpc = cost.RRt_mpc
    QQT_mpc = cost.QQT_mpc

    AA = np.zeros((params.ns, params.ns, TT))
    BB = np.zeros((params.ns, params.nu, TT))

    # 1. Linearize the dynamics about the optimal trajectory
    for tt in range(TT-1):
        fx, fu = dyn.dynamics(xx_opt[:,tt], uu_opt[:,tt])[1:]  # Linearization
        AA[:,:,tt] = fx.T
        BB[:,:,tt] = fu.T

    # 2. Initialize state and control input trajectories
    xx_real = np.tile(xx_opt[:,0], (TT, 1)).T
    uu_real = np.tile(uu_opt[:,0], (TT, 1)).T

    # Introduce a perturbation to show tracking properties of the MPC
    xx_real[:, 0] = xx_opt[:,0].squeeze() + perturbation_MPC

    JJ = 0  # Cost function value
    JJ_mpc_list = []
    counter_total = 0  # Counter for infeasible problems

    # 3. Execute the outer loop of the MPC algorithm
    for tt in range(T_sim-1):
        xx_t_mpc = xx_real[:,tt]
        if tt%5 == 0:
            print(f"Solving the MPC: Iteration {tt}")

        # 4. Solve the MPC problem by calling the inner loop
        uu_real[:,tt], cst, counter = MPC_solve_step(AA[:,:,tt:], BB[:,:,tt:], QQt_mpc, RRt_mpc, QQT_mpc, xx_t_mpc, xx_opt[:,tt:], uu_opt[:,tt:], x1_min, x1_max, x2_min, x2_max, u1_min, u1_max, flag_constr_input_MPC, T_pred=T_pred_mpc)
        counter_total += counter
        if uu_real is None or np.any(np.isnan(uu_real)):
            print(f"[WARN] Iter {tt}: not valid control input, using previous one.")
            uu_real[:,tt] = uu_real[:,tt-1] if tt > 0 else np.zeros((nu,))
            xx_real[:,tt+1] = xx_real[:,tt]
            continue
        
        xx_real[:,tt+1] = dyn.dynamics(xx_real[:,tt], uu_real[:,tt])[0].flatten()  # Propagate the state

        # Extra perturbation to show tracking properties of the MPC
        if extra == True and tt == T_sim//2: 
            print(colored(f"Extra perturbation at time step {tt}, whit value {perturbation_MPC}", 'yellow', attrs=['bold']))
            xx_real[:,tt] += perturbation_MPC

        # 5. For plot purposes, keep the last control input and state constant
        uu_real[:,-1] = uu_real[:,-2]
        xx_real[:,-1] = xx_real[:,-2]
        JJ += cst
        JJ_mpc_list.append(cst)

    print(colored(f"Number of worning constraints: {(counter_total)}", 'yellow', attrs=['bold']))

    return xx_real, uu_real, JJ_mpc_list
