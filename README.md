# Optimal Control of a Flexible Robotic Arm

![Python](https://img.shields.io/badge/Python-3.10-blue)


This work addresses the design of optimal control for a flexible roboticarm, a system representative of high-precision applications such as medical robotics and industrial automation. The considered model describes a planar two-link robot with torque applied only at the first joint, incorporating nonlinear forces, viscous friction, and gravity effects. The project follows a structured approach: first, the system’s dynamics are discretized, and two equilibrium states are identified. An optimal transition between these states is then computed using Newton-based numerical op timization. To ensure accurate trajectory tracking, even in the presence of disturbances, advanced control techniques such as the Linear-Quadratic Regulator (LQR) and Model Predictive Control (MPC) are imple mented. This study demonstrates the effectiveness of optimal control strategies in handling flexible robotic systems, emphasizing their potential for real-world applications

---
---

## CODE USE GUIDE


This document provides a guide on how to use the provided code. 


Open the folder `Optimal_Control_of_a_Flexible-Robotic-Arm` in the terminal, and run: 

```bash
python code/main.py
```

## Or follow the Recommended Installation & Setup

### Clone the repository 

 ```bash
git clone <repository-url>

cd Optimal_Control_of_a_Flexible-Robotic-Arm
```

### Create a virtual environment
 ```bash
python -m venv venv
```

*Activate it*:

Windows

**venv\Scripts\activate**

Mac/Linux

**source venv/bin/activate**

### Install dependencies
 ```bash
pip install -r requirements.txt
```

### Run the project
 ```bash
python code/main.py
```


---

### REPORT FOLDER  

In this folder we can finde an PDF and Latex file where there is an all description on the project. 

Principally the project goal is distribuite by: 
* **Task_1**: where simply *step* trajectory are implement and optimize by the Newthons' method 
* **Task_2** we implemet different *smooth* trajectory and optimize by same method
* **Task_3**: when we obtain an *Optimization Smooth Curve* for Task_2 we are implemet LQR 
* **Task_4**:  when we obtain an *Optimization Smooth Curve* for Task_2 we are implemet MPC and add also an comparison to both methods 
* **Task_5**: Animate the pendolum for all task 

---

### LIST OF FILES IN CODE FOLDER 

* **`main.py`**
  Main script to run the simulation and control algorithms.

* **`flex_arm_parameters.py`**
  Contains all simulation and model parameters.

* **`flex_arm_trajectory_generation.py`**
  Functions for generating reference trajectories and performing trajectory optimization.

* **`flex_arm_trajectory_tracking.py`**
  Implements the Model Predictive Control (MPC) and trajectory tracking routines.

* **`flex_arm_cost.py`**
  Cost function definitions for optimization and control.

* **`flex_arm_animation.py`**
  Animation utilities for visualizing the robotic arm’s motion.

* **`flex_arm_dynamics.py`**
  Contains the dynamic model and equations of motion for the flexible arm.

* **`flex_arm_plot.py`** 
  Plotting utilities for results and diagnostics.

* **`simulation_data_handler.py`**
  Functions for saving and loading simulation data.

Additionally, within the folder, you will find a subfolder named `Simulation_Data` where the simulation results for Task_1 and Task_2 are stored.

---
Optimal_Control_of_a_Flexible-Robotic-Arm/
│
├── code/
│   ├── main.py
│   ├── flex_arm_parameters.py
│   ├── flex_arm_dynamics.py
│   ├── flex_arm_trajectory_generation.py
│   ├── flex_arm_trajectory_tracking.py
│   ├── flex_arm_animation.py
│   ├── flex_arm_cost.py
│   ├── flex_arm_plot.py
│   └── simulation_data_handler.py
│
├── Animation/
├── report/
└── requirements.txt

---
---
### PARAMETER TO COSTUMAZI THE SIMULATION 


The file `flex_arm_parameters.py` contains all the main parameters for the simulation and control of the flexible robotic arm. Below is a summary of the key parameters and their roles:



#### Configuration Flags

The Boolean flags that control the behavior and visualization of the simulation. You can set these flags to `True` or `False` to enable or disable specific features:

* `flag_plot_ref`: Plot the reference trajectory.
* `state_variable`: Plot the state variables of the system.
* `flag_plot_armijo`: Plot the Armijo descent during optimization.
* `flag_plot_descent`: Plot the descent curve of the optimization algorithm.
* `flag_anim`: Show an animation of the pendulum/robotic arm.
* `flag_error_ref`: Plot the error between the reference and the optimal trajectory.
* `flag_plot_vel`: Plot the velocity of the pendulum/robotic arm.
* `armijo_loop`: If `True`, use closed-loop Armijo; if `False`, use open-loop Armijo.
* `print_debug`: Print debug information during execution.
* `simulation_review`: If `True`, load simulation data from file instead of running a new simulation.
* `flag_MPC_difference_constr`: Show the difference between MPC with and without input constraints.

---

## Main Simulation Parameters

The file `flex_arm_parameters.py` contains all the main parameters for the simulation and control of the flexible robotic arm. Below is a summary of the key parameters and their roles:

### Simulation Parameters

- **dt**: Discretization step (time step for integration), e.g., `1e-2`.
- **ns**: Number of states in the system (e.g., 4).
- **nu**: Number of control inputs (e.g., 1).
- **tf**: Total simulation time (seconds).
- **TT**: Number of time samples (`TT = int(tf / dt)`).
- **maxiters**: Maximum number of iterations for Newton's method.

---

### Equilibrium States

The system can be initialized at different equilibrium points, defined as follows:

- **xx_eq_1**: First equilibrium state (e.g., θ₁ = 210°, θ₂ = -210°, both velocities zero).
- **xx_eq_2**: Second equilibrium state (e.g., θ₁ = 30°, θ₂ = -30°, both velocities zero).
- **xx_eq_3**: Third equilibrium state (θ₁ = 0°, θ₂ = 0°, both velocities zero) – not used by default.
- **xx_eq_4**: Fourth equilibrium state (θ₁ = 45°, θ₂ = -45°, both velocities zero) – not used by default.

All angles are in radians.

---

### MPC Time Parameters and Constraints

- **T_pred_mpc**: Prediction horizon for the Model Predictive Control (MPC).
- **T_sim_mpc**: Simulation time for the MPC (usually equal to `TT`).

#### State and Input Constraints

- **x1_max, x1_min**: Maximum and minimum values for state variable x1.
- **x2_max, x2_min**: Maximum and minimum values for state variable x2.
- **x1_vel, x2_vel**: Maximum and minimum velocities for x1 and x2.
- **u1_max, u1_min**: Maximum and minimum values for the control input u1.

---

### Initial State Perturbations

To test the robustness of the controllers, several perturbations can be applied to the initial state:

- **perturbation_LQR_small / medium / large**: Small, medium, and large perturbations for LQR tests.
- **perturbation_MPC_small / medium / large**: Small, medium, and large perturbations for MPC tests.

All perturbations are defined as NumPy arrays of length 4 (matching the state dimension).

---


You can modify the parameters in `flex_arm_parameters.py` to customize the simulation according to your needs.

Additionally, in `main.py` you can configure:

* The task runner behavior, which follows the main control policy;
* The reference trajectory used during the simulation, by changing the parameter `type_ref_traj_T1` for Task 1 and `type_ref_traj_T2` for Task 2;
* The type of perturbation applied during the simulation for both LQR and MPC controllers.

---
---

# Experimental Results – 3-Point Smooth Trajectory Tracking

This section presents the tracking performance of the flexible robotic arm when following a 3-point smooth cubic spline reference trajectory under different disturbance conditions.

The comparison involves:

- Task 2 – Smooth optimal trajectory via Newton’s Method 

- Task 3 – Trajectory tracking using LQR

- Task 4 – Trajectory tracking using MPC

Each experiment shows the animation of the pendulum motion.

--- 
## Experiment 1 – No Perturbation

The system starts exactly from the reference initial condition.

<div align="center">
Task 2 – Newton’s Method 	Task 3 – LQR	Task 4 – MPC
<img src="Animation\cs_w3p\None\T2\Gif_t2_with_none.gif" width="260">	<img src="Animation\cs_w3p\None\T3\Gif_t3_with_none.gif" width="260">	<img src="Animation\cs_w4p\None\T4\Gif_t4_with_none.gif" width="260">
</div>

--- 
## Experiment 2 – Initial State Perturbation

A perturbation is applied to the initial state to test robustness.

<div align="center">
Task 2 – Newton’s Method	Task 3 – LQR	Task 4 – MPC
<img src="Animation\cs_w3p\Pertubation\T2\Gif_t2_with_pert.gif" width="260">	<img src="Animation\cs_w3p\Pertubation\T3\Gif_t3_with_pert.gif" width="260">	<img src="Animation\cs_w3p\Pertubation\T4\Gif_t4_with_pert.gif" width="260">
</div>

---
## Experiment 3 – Mid-Simulation Perturbation

An additional disturbance is injected halfway through the simulation.

<div align="center">
Task 2 – Newton’s Method	Task 3 – LQR	Task 4 – MPC
<img src="Animation\cs_w3p\Extra_mid\T2\Gif_t2_with_extra.gif" width="260">	<img src="Animation\cs_w3p\Extra_mid\T2\Gif_t3_with_extra.gif" width="260">	<img src="Animation\cs_w3p\Extra_mid\T2\Gif_t4_with_extra.gif" width="260">
</div>