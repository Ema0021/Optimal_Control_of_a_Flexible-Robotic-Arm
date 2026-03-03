import numpy as np
from scipy.interpolate import CubicSpline
from termcolor import colored

import flex_arm_parameters as params
import flex_arm_compute_equilibria as compeq
import flex_arm_dynamics as dyn
import matplotlib.pyplot as plt
import flex_arm_cost as cst
import plot as plot


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Plotting settings
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 8})

ns = params.ns                          # number of states
nu = params.nu                          # number of input
TT = params.TT                          # time samples
TT_half = int(TT/2)                     # T/2 when there's the transition
t = np.linspace(0, params.tf, TT)
maxiters = params.maxiters              # iterations of Newton's method
term_cond = 1e-6                        # value for the terminal condition of Newton's method

# Armijo Parameters
stepsize_0 = 1.0                        # initial stepsize
cc = 0.5                                # 0<c<1 - Const of Slope of Armijo line search
beta = 0.7                              # reduction factor of the stepsize
armijo_maxiters = 20                    # number of Armijo iterations

# Debugging flag
print_debug = params.print_debug                # set to True to print debug information

# Plotting flags
flag_plot_descent = params.flag_plot_descent    # set to True to plot the descent direction
flag_plot_ref = params.flag_plot_ref            # set to True to plot the reference trajectory
flag_plot_armijo = params.flag_plot_armijo      # set to True to plot the Armijo descent

# Simulation review flag
simulation_review = params.simulation_review 

###################################
# REFERENCE CURVE GENERATION      #
###################################

def reference_curve(xx_eq1, uu_eq1, xx_eq2, uu_eq2, type_ref_traj,flag_plot_ref=True):
    """
    Define and generate a reference curves for a pendulum based on the given equilibrium points 
    and type of reference trajectory.
    
    
    Parameters:
        - xx_eq1: State vector at equilibrium point 1.
        - uu_eq1: Input vector at equilibrium point 1.
        - xx_eq2: State vector at equilibrium point 2.
        - uu_eq2: Input vector at equilibrium point 2.
        - type_ref_traj: Type of trajectory (e.g. "step", "cs", "cs_w3p", "cs_w7p" ).
        - flag_plot_ref: Boolean flag to indicate whether to plot the reference trajectory.

    Returns:
        - xx_ref: Generated state reference trajectory.
        - uu_ref: Generated input reference trajectory.

    """

    # Convert to numpy arrays
    xx_eq1 = np.array(xx_eq1)
    uu_eq1 = np.array(uu_eq1)
    xx_eq2 = np.array(xx_eq2)
    uu_eq2 = np.array(uu_eq2)
   
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((nu, TT))

    if type_ref_traj == "step":

        # Step Trajectory
        xx_ref[:, :TT_half] = np.tile(xx_eq1.reshape(-1, 1), TT_half)
        xx_ref[:, TT_half:] = np.tile(xx_eq2.reshape(-1, 1), TT_half)
        uu_ref[:, :TT_half] = np.tile(uu_eq1.reshape(-1, 1), TT_half)
        uu_ref[:, TT_half:] = np.tile(uu_eq2.reshape(-1, 1), TT_half)       

    elif type_ref_traj == "cs":

        # Cubic Spline Trajectory
        tao = int(0.2 * TT)
        tt_ref = np.array([TT_half - tao, TT_half + tao])  # Keypoints for spline
        cs_xx = CubicSpline(tt_ref, np.vstack([xx_eq1, xx_eq2]), bc_type='clamped')
        cs_uu = CubicSpline(tt_ref, np.vstack([uu_eq1, uu_eq2]), bc_type='clamped')
        for tt in range(TT):
            if tt < TT_half - tao:
                xx_ref[:, tt] = xx_eq1
                uu_ref[:, tt] = uu_eq1
            elif tt > TT_half + tao:
                xx_ref[:, tt] = xx_eq2
                uu_ref[:, tt] = uu_eq2
            else:
                xx_ref[:, tt] = cs_xx(tt)
                uu_ref[:, tt] = cs_uu(tt)
        
    elif type_ref_traj == 'cs_w3p': 

        # Cubic Splinr whit different point 
        tt_ref = np.array([int(0.2*TT), TT_half, int(0.8*TT)])  # Keypoints for spline
        xx_points = np.column_stack([
            xx_eq1,
            (xx_eq1 + xx_eq2) / 2,  # state intermidiate
            xx_eq2
        ])
        uu_points = np.column_stack([
            uu_eq1,
            (uu_eq1 + uu_eq2) / 2,  # input intermidiate
            uu_eq2
        ])

        # Create cubic spline for each component
        cs_xx = CubicSpline(tt_ref, xx_points.T, bc_type='clamped')
        cs_uu = CubicSpline(tt_ref, uu_points.T, bc_type='clamped')

        for tt in range(TT):
            if tt < tt_ref[0]:
                xx_ref[:, tt] = xx_points[:, 0]
                uu_ref[:, tt] = uu_points[:, 0]
            elif tt > tt_ref[-1]:
                xx_ref[:, tt] = xx_points[:, -1]
                uu_ref[:, tt] = uu_points[:, -1]
            else:
                xx_ref[:, tt] = cs_xx(tt)
                uu_ref[:, tt] = cs_uu(tt)
        

    elif type_ref_traj == 'cs_w7p':

        # Cubic Spline with 7 points
        n_eq = 7
        tt_ref = np.array([int(0.2*TT), int(0.4*TT), int(0.45*TT), TT_half, int(0.55*TT), int(0.7*TT), int(0.8*TT)])  # Keypoints for spline

        # Generate 7 equilibrium states
        xx_points = np.column_stack([
            (1 - alpha) * xx_eq1 + alpha * xx_eq2 
            for alpha in [0.0, 0.2, 0.05, 1.0, 0.8, 0.3, 0.0]  
        ])
        uu_points = np.column_stack([
            (1 - beta) * uu_eq1 + beta * uu_eq2
            for beta in [0.0, 0.2, 0.05, 1.0, 0.8, 0.3, 0.0]
        ])

        # Create cubic spline for each component
        cs_xx = CubicSpline(tt_ref, xx_points.T, bc_type='clamped')
        cs_uu = CubicSpline(tt_ref, uu_points.T, bc_type='clamped')

        for tt in range(TT):
            if tt < tt_ref[0]:
                xx_ref[:, tt] = xx_points[:, 0]
                uu_ref[:, tt] = uu_points[:, 0]
            elif tt > tt_ref[-1]:
                xx_ref[:, tt] = xx_points[:, -1]
                uu_ref[:, tt] = uu_points[:, -1]
            else:
                xx_ref[:, tt] = cs_xx(tt)
                uu_ref[:, tt] = cs_uu(tt)

    
    return xx_ref, uu_ref, type_ref_traj, ns, t



def traj_gen_newton(xx_ref, uu_ref):
    
    """
    Generates the optimal trajectory using Newton's algorithm.

    Parameters:
        - xx_ref (numpy.ndarray): Reference state trajectory.
        - uu_ref (numpy.ndarray): Reference control trajectory.
        
    Returns:
        - xx_opt (numpy.ndarray): Optimal state trajectory.
        - uu_opt (numpy.ndarray): Optimal control trajectory.
        - JJ (numpy.ndarray): Cost at each iteration of the Newton's algorithm.
        - xx_plot (numpy.ndarray): State trajectory at each iteration of the Newton's algorithm.
        - uu_plot (numpy.ndarray): Control trajectory at each iteration of the Newton's algorithm.
        - descent (numpy.ndarray): Descent direction at each iteration of the Newton's algorithm.
        - KK (numpy.ndarray): Gain matrix at each time step.

    """
    ######################################
    # ARRAYS TO STORE DATA               #
    ######################################

    # Declaretion of all needed quantities for Newton's algorithm 
    AA = np.zeros((ns,ns,TT)) 
    BB = np.zeros((ns,nu,TT))
    QQ = np.zeros((ns,ns,TT))
    RR = np.zeros((nu,nu,TT))
    QQT = cst.QQT      
    qq = np.zeros((ns,TT))
    rr = np.zeros((nu,TT))
    qqT = np.zeros(ns)
    KK = np.zeros((nu, ns, TT))
    sigma = np.zeros((nu, TT))
    PP = np.zeros((ns, ns, TT))
    pp = np.zeros((ns, TT))
    lmbd = np.zeros((ns, TT, maxiters))           
    dJ = np.zeros((nu,TT, maxiters))                # gradient of J wrt u

    JJ = np.zeros(maxiters)                         # collect cost = stage_cost + terminal_cost
    descent = np.zeros(maxiters)                    # collect descent direction
    descent_arm = np.zeros(maxiters)                # collect descent direction

    # Initialize state and control trajectories with reference values 
    xx_opt = np.tile(xx_ref[:,0], (TT,1)).T         
    uu_opt = np.tile(uu_ref[:,0], (TT,1)).T
    xx_opt_kprec = np.tile(xx_ref[:, 0], (TT,1)).T
    xx_plot = np.zeros((ns,TT, maxiters))
    uu_plot = np.zeros((nu,TT, maxiters))

    if print_debug == True:
        print("KK shape", KK.shape)
        print("Signa schape",sigma.shape)
        print("xx_ref shape", xx_ref.shape)
        print("uu_ref shape", uu_ref.shape)
        print("xx_opt shape", xx_opt.shape)
        print("uu_opt shape", uu_opt.shape)
        print("xx_opt_kprec shape", xx_opt_kprec.shape)
        print("QQT shape", QQT.shape)
        print("QQ shape", QQ.shape)
        print("RR shape", RR.shape)
        print("AA shape", AA.shape)
        print("BB shape", BB.shape)

 

    ###########################################################################
    # Iterate over the maximum number of iterations of the Newton's algorithm #
    ###########################################################################

    for kk in range(maxiters): 
        # Compute total cost: sum of stage costs and terminal cost
        JJ[kk] = 0
        term_cost, qqT = cst.terminalcost(xx_opt[:,-1], xx_ref[:,-1], QQT)
        JJ[kk] += term_cost

        for tt in range(TT-1):
            _, dfx,dfu = dyn.dynamics(xx_opt[:,tt], uu_opt[:,tt])
            stage_cost, qqt,rrt = cst.stagecost(xx_opt[:,tt], uu_opt[:,tt], xx_ref[:,tt], uu_ref[:,tt])
            JJ[kk] += stage_cost

            # Filling matrices
            AA[:,:,tt] = dfx.T      #A_t
            BB[:,:,tt] = dfu.T      #B_t
            qq[:,tt] = qqt          #q_t = dlt wrt x 
            rr[:,tt] = rrt          #r_t = dlt wrt u
            QQ[:,:,tt] = cst.QQt    #Q_t
            RR[:,:,tt] = cst.RRt    #R_t

        ##################################
        # COMPUTATION OF K AND SIGMA     #
        ################################## 
        PP[:,:,-1] = QQT                   #P_T = Q_T
        pp[:,-1] = qqT                     #p_T = q_T

        for tt in reversed(range(TT-1)):    #(from TT-2 to 0 included)
            QQt = QQ[:,:,tt]
            qqt = qq[:,tt][:,None]
            RRt = RR[:,:,tt]
            rrt = rr[:,tt][:,None]
            AAt = AA[:,:,tt]
            BBt = BB[:,:,tt]
            PPtp = PP[:,:,tt+1]             #P_{t+1}    (updating value)
            pptp = pp[:, tt+1][:,None]      #p_{t+1}

            MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
            mmt = rrt + BBt.T @ pptp

            KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt) 
            sigma_t = -MMt_inv @ mmt
            ppt = qqt + AAt.T @ pptp - KK[:,:,tt].T @ (RRt + BBt.T @ PPtp @ BBt) @ sigma_t        
            PPt = QQt + AAt.T @ PPtp @ AAt - KK[:,:,tt].T @ (RRt + BBt.T @ PPtp @ BBt) @ KK[:,:,tt]

            PP[:,:,tt] = PPt
            pp[:,tt] = ppt.squeeze()
            sigma[:,tt] = sigma_t.squeeze()

        ##################################
        #     ARMIJO IMPLEMENTATION      #
        ################################## 

        ##################################
        # DESCENT DIRECTION CALCULATION  #
        ##################################

        lmbd_temp = cst.terminalcost(xx_opt[:,TT-1], xx_ref[:,TT-1], QQT)[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

        delta_x = np.zeros((ns, TT))
        delta_u = np.zeros((nu, TT))

        for tt in reversed(range(TT-1)):
            # Computing lambdas
            lmbd_temp = AA[:,:,tt].T @ lmbd[:,tt+1,kk] + qq[:,tt]       # co-state equation
            dJ_temp = BB[:,:,tt].T @ lmbd[:,tt+1,kk] + rr[:,tt]         # gradient of J wrt u

            # Filling matrices
            lmbd[:,tt,kk] = lmbd_temp.squeeze()     # lambda_t_k
            #print("lmbd ", lmbd[:,tt,kk], "at t", tt, ", k", kk)
            dJ[:,tt,kk] = dJ_temp.squeeze() 
            #print("Lambda at ", kk, "-th k iteration: ", lmbd[:,tt,kk], "tt: ", tt)       
        
        for tt in range(TT-1): 
            delta_u[:,tt] = KK[:, :, tt] @ delta_x[:,tt] + sigma[:, tt] 
            delta_x[:,tt+1] = AA[:,:,tt] @ delta_x[:,tt] + BB[:,:,tt] @ delta_u[:,tt] 
            descent[kk] += delta_u[:,tt].T @ delta_u[:,tt] 
            descent_arm[kk] += dJ[:,tt,kk].T @ delta_u[:,tt] 

            if print_debug == True:

                print("Delta u at ", kk, "-th k iteration: ", delta_u[:,tt], " - t: ", tt) 

        #######################################
        # Stepsize selection with Armijo Rule #
        #######################################

        stepsizes = []           # gamma list containing gamma_k
        costs_armijo = []        # list containing J[kk]

        stepsize = stepsize_0
        print(colored(f"\n -------START Iter. {kk}  FOR ARMIJO---------", "blue", attrs=["bold", "underline"]))
        # Perform Armijo line search to determine appropriate step size
        for ii in range(armijo_maxiters):
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((nu,TT))

            xx_open = np.zeros((ns,TT))
            xx_closed = np.zeros((ns,TT))

            xx_open[:,0] = xx_opt[:,0] 
            xx_closed[:,0] = xx_ref[:,0] 

            if print_debug == True:
                print("xx_open[:,0] ", xx_open[:,0], "at t: ", 0)
                print("xx_closed[:,0] ", xx_closed[:,0], "at t: ", 0)
            
            if params.armijo_loop == False:
                # Open loop Armijo
                xx_temp[:,0] = xx_ref[:,0]
                
                for tt in range(TT-1):
                    uu_temp[:,tt] = uu_opt[:,tt] + stepsize*delta_u[:,tt]
                    # print("xx_temp{t} - armijo : ", xx_temp[:,tt], "at tt: ", tt)
                    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()
            else :
                # Closed loop Armijo
                xx_temp[:,0] = xx_ref[:,0]  # non xx_ref[:,0]

                for tt in range(TT-1):
                    uu_temp[:,tt] = uu_opt[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx_opt[:,tt]) + stepsize * sigma[:, tt]
                    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()

            # Tempcost computation
            JJ_temp = 0
            for tt in range(TT-1):
                JJ_temp += cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += cst.terminalcost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]

            stepsizes.append(stepsize)
            costs_armijo.append(JJ_temp)


            print("Armijo iteration: ", ii, " - stepsize: ", stepsize, " - JJ_temp: ", JJ_temp)
            if JJ_temp > JJ[kk] + cc*stepsize*descent_arm[kk]:
                stepsize = beta*stepsize
                print("Armijo stepsize updated. It becomes stepsize = {} \n ".format(stepsize))
            else:
                print("Armijo stepsize not updated. It remains stepsize = {} \n ".format(stepsize))
                break
            if ii == armijo_maxiters -1:
                print("WARNING: NO stepsize was found with armijo rule!!! \n ")


        ##################################
        #           PLOT ARMIJO          #
        ##################################

        # Plot Armijo descent if flag is set
        if flag_plot_armijo:
            
            steps = np.linspace(0, stepsize_0, armijo_maxiters) # 20 steps between 0 and stepsize_0 int(2e1)

        
            print(colored("\n -------PLOT ARMIJO----------", "blue", attrs=["bold", "underline"]))

            if print_debug == True:
                print(colored(f"steps:  {steps}", "blue"))
                print(colored(f"Length steps: {len(steps)}", "blue"))

            costs = np.zeros(len(steps))
    
            # Calculate costs for each step
            for ii in range(len(steps)):
    
                step = steps[ii]
    
                # temp solution update
    
                xx_temp = np.zeros((ns,TT))
                uu_temp = np.zeros((nu,TT))
                
                if params.armijo_loop == False:
                    # Open loop Armijo
                    xx_temp[:,0] = xx_ref[:,0]      #initial state - x0

                    print("plot _Armijo, iteration i: ", ii)

                    for tt in range(TT-1):

                        uu_temp[:,tt] = uu_opt[:,tt] + step * delta_u[:,tt]
                        xx_temp_plus = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]
                        xx_temp[:,tt+1] = xx_temp_plus.squeeze()
                
                else:
                    # Closed loop Armijo
                    xx_temp[:,0] = xx_opt[:,0]
                    
                    print("plot _Armijo, iteration i: ", ii)

                    for tt in range(TT-1):
                        uu_temp[:,tt] = uu_opt[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx_opt[:,tt]) + step * sigma[:, tt]
                        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()
                    
                JJ_temp = 0
    
                for tt in range(TT-1):
                    temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost
    
                temp_cost = cst.terminalcost(xx_temp[:,-1], xx_ref[:,-1], cst.QQT)[0]
                JJ_temp += temp_cost
    
                costs[ii] = JJ_temp
 
            plt.figure(figsize=(10, 6))
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - \\text{stepsize} \\cdot d^k)$')
            plt.plot(steps, JJ[kk] + descent_arm[kk] * steps, color='r', label='$J(\\mathbf{u}^k) - \\text{stepsize} \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ[kk] + cc * descent_arm[kk] * steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - \\text{stepsize} \\cdot c \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    
            # Scatter the tested step sizes
            plt.scatter(stepsizes, costs_armijo, marker='*', label="Tested Stepsizes")
        
            plt.xlabel('Step Size', fontsize=14)
            plt.ylabel('Cost', fontsize=14)
            plt.title(f"Armijo stepsize selection: Iteration {kk}")
            plt.legend()
            plt.grid()
            plt.show()
            

        
        ##########################################
        # Evaluation of the optimal trajectories #
        ########################################## 

        # Initialize the optimal trajectories with the reference values
        xx_opt[:, 0] = xx_ref[:, 0]             # x_opt is xx_ref[:, 0]
        xx_opt_kprec[:, 0] = xx_ref[:, 0]       # xx_opt_kprec now it's a 1D (4,) array for the computation

        # Update control inputs using computed feedback gains and step size
        for tt in range(TT-1):
            
            uu_opt[:, tt] = uu_opt[:, tt] + KK[:, :, tt] @ (xx_opt[:, tt] - xx_opt_kprec[:, tt]) + stepsize * sigma[:, tt]
            xx_opt[:, tt+1] = dyn.dynamics(xx_opt[:, tt], uu_opt[:, tt])[0].squeeze()
            xx_opt_kprec[:, tt] = xx_opt[:, tt]
            
        # Debugging information
        if print_debug == True:
            print("Initial uu_opt shape:", uu_opt.shape)
            print("xx_opt_kprec shape:\n ", xx_opt_kprec.shape)  

        print(f"\n [Iteration {kk}] Cost: {JJ[kk]} \t -- Norm of descent direction: {descent[kk]} \n")

        xx_plot[:,:,kk] = xx_opt
        uu_plot[:,:,kk] = uu_opt

        # Stopping criterion
        if np.isclose(descent[kk], 1e-7, atol=term_cond):
            xx_plot = xx_plot[:,:,:kk+1]
            uu_plot = uu_plot[:,:, :kk+1]
            break

    # Plot descent direction
    if flag_plot_descent and simulation_review == False:
        print(colored("\n ------------PLOT DESCENT DIRECTION------------\n", "blue", attrs=["bold", "underline"]))
        plot.semilog_descent_plot(descent)



    return xx_opt, uu_opt, JJ, xx_plot, uu_plot, descent, kk



