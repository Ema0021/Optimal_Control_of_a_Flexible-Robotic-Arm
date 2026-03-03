from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

def animate_double_pendulum(xx_star, xx_ref, title, dt, save_frames = False):
    """
    Animates the double pendulum dynamics
    input parameters:
        - xx_star: optimal state trajectory (including angles for both pendulums)
        - xx_ref: reference trajectory (including angles for both pendulums)
        - title: title for the plot
        - dt: Sampling time
        - save_frames: whether to save keyframes of the animation
    output arguments:
        None
    """

    TT = xx_star.shape[1]
    
    # Set up the figure and axis for the animation
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Plot elements
    pendulum_line = ax.plot([], [], 'o-', lw=3, color="blue", label="Optimal Path")[0]
    reference_line = ax.plot([], [], 'o--', lw=2, color="green", label="Reference Path")[0]
    trail_opt_line = ax.plot([], [], '-', lw=1, color='blue', alpha=0.5, label="Optimal Trail")[0]
    trail_ref_line = ax.plot([], [], '--', lw=1, color='green', alpha=0.5, label="Reference Trail")[0]

    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.legend()
    ax.set_title("Double Pendulum Optimal Control Trajectory - " + title)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    # Storage for trails
    trail_opt_x, trail_opt_y = [], []
    trail_ref_x, trail_ref_y = [], []

    def init():
        pendulum_line.set_data([], [])
        reference_line.set_data([], [])
        trail_opt_line.set_data([], [])
        trail_ref_line.set_data([], [])
        time_text.set_text('')
        return pendulum_line, reference_line, trail_opt_line, trail_ref_line, time_text

    def update(frame):

        # Optimal positions for pendulum 1 (theta1) and pendulum 2 (theta2)
        x_opt1 = np.sin(xx_star[0, frame])
        y_opt1 = -np.cos(xx_star[0, frame])
        x_opt2 = x_opt1 + np.sin(xx_star[0, frame] + xx_star[1, frame])
        y_opt2 = y_opt1 - np.cos(xx_star[0, frame] + xx_star[1, frame])

        # Reference positions for pendulum 1 (theta1) and pendulum 2 (theta2)
        x_ref1 = np.sin(xx_ref[0, frame]) # assuming xx_ref[0] is angle theta1
        y_ref1 = -np.cos(xx_ref[0, frame])
        x_ref2 = x_ref1 + np.sin(xx_ref[0, frame] + xx_ref[1, frame]) # assuming xx_ref[1] is angle theta2
        y_ref2 = y_ref1 - np.cos(xx_ref[0, frame] + xx_ref[1, frame])

        # Update pendulum line (optimal)
        pendulum_line.set_data([0, x_opt1, x_opt2], [0, y_opt1, y_opt2])
        # Update reference line
        reference_line.set_data([0, x_ref1, x_ref2], [0, y_ref1, y_ref2])
        # Update time text
        time_text.set_text(f'time = {frame*dt:.2f}s')

        # Update trails
        trail_opt_x.append(x_opt2)
        trail_opt_y.append(y_opt2)
        trail_ref_x.append(x_ref2)
        trail_ref_y.append(y_ref2)
        trail_opt_line.set_data(trail_opt_x, trail_opt_y)
        trail_ref_line.set_data(trail_ref_x, trail_ref_y)

        return pendulum_line, reference_line, trail_opt_line, trail_ref_line, time_text

    ani = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=1000*dt)

    plt.show()

    if save_frames:
        # Clear trails for save the animation
        trail_opt_x.clear()
        trail_opt_y.clear()
        trail_ref_x.clear()
        trail_ref_y.clear()

        # Frame to save
        keyframes = {
            0: f"[OPTCON2024]-Group32/code/Animation/Test/pendulum_init_{title}.png",
            TT // 2: f"[OPTCON2024]-Group32/code/Animation/Test/pendulum_mid_{title}.png",
            TT - 1: f"[OPTCON2024]-Group32/code/Animation/Test/pendulum_final_{title}.png"
        }

        # Generate frames with trails
        for frame_idx, filename in keyframes.items():
            trail_opt_x.clear()
            trail_opt_y.clear()
            trail_ref_x.clear()
            trail_ref_y.clear()
            
            # Add the trials for the current frame
            for f in range(frame_idx + 1):
                update(f)
            
            # Save the figure
            fig.savefig(filename)
            print(f"Save in: {filename}")