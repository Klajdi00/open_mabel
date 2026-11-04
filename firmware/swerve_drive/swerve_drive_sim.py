import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. Define Robot Geometry (3-Wheel Triangular Base)
# ---------------------------------------------------------
# Assume +Y is Forward, +X is Right
# Wheel coordinates (x, y) relative to the center of the robot
CHASSIS_RADIUS = 1.0

# W1: Front Right, W2: Front Left, W3: Back Center
wheel_x = np.array([0.866, -0.866, 0.0]) * CHASSIS_RADIUS
wheel_y = np.array([0.5, 0.5, -1.0]) * CHASSIS_RADIUS

# ---------------------------------------------------------
# 2. Kinematics Function
# ---------------------------------------------------------
def calculate_swerve_vectors(v_x, v_y, omega):
    """
    v_x: Strafe velocity (positive = right)
    v_y: Forward velocity (positive = forward)
    omega: Rotational velocity (positive = counter-clockwise)
    """
    # Translation vector is the same for all wheels
    u_trans = v_x
    v_trans = v_y
    
    # Rotation vector = omega x r = (-omega * y, omega * x)
    u_rot = -omega * wheel_y
    v_rot = omega * wheel_x
    
    # Combined wheel velocity vectors
    u_total = u_trans + u_rot
    v_total = v_trans + v_rot
    
    return u_total, v_total

# ---------------------------------------------------------
# 3. Plot Setup
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3) # Make room for sliders
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("3-Wheel Swerve Kinematics Simulator", fontsize=14)
ax.set_xlabel("+X (Strafe Right)")
ax.set_ylabel("+Y (Drive Forward)")
ax.grid(True, linestyle='--', alpha=0.6)

# Draw Chassis (Triangle)
chassis = patches.Polygon(np.column_stack([wheel_x, wheel_y]), closed=True, 
                          fill=True, color='lightgray', alpha=0.5, edgecolor='black', linewidth=2)
ax.add_patch(chassis)

# Draw Wheels
ax.scatter(wheel_x, wheel_y, color='black', s=100, zorder=5, label='Swerve Modules')

# Initial Vectors (0 velocity)
u_init, v_init = calculate_swerve_vectors(0, 0, 0)
quiver = ax.quiver(wheel_x, wheel_y, u_init, v_init, 
                   color='red', scale=1, scale_units='xy', angles='xy', 
                   width=0.015, zorder=10, label='Target Velocity Vector')

ax.legend(loc='upper right')

# ---------------------------------------------------------
# 4. Interactive Sliders Setup
# ---------------------------------------------------------
axcolor = 'lightgoldenrodyellow'
ax_vy = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_vx = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_omega = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

s_vy = Slider(ax_vy, 'Forward (Vy)', -2.0, 2.0, valinit=0.0)
s_vx = Slider(ax_vx, 'Strafe (Vx)', -2.0, 2.0, valinit=0.0)
s_omega = Slider(ax_omega, 'Rotate (Omega)', -2.0, 2.0, valinit=0.0)

# ---------------------------------------------------------
# 5. Update Logic
# ---------------------------------------------------------
def update(val):
    # Read current slider values
    vx = s_vx.val
    vy = s_vy.val
    omega = s_omega.val
    
    # Calculate new vectors
    u, v = calculate_swerve_vectors(vx, vy, omega)
    
    # Update quiver plot
    quiver.set_UVC(u, v)
    
    # Redraw canvas
    fig.canvas.draw_idle()

# Link sliders to the update function
s_vy.on_changed(update)
s_vx.on_changed(update)
s_omega.on_changed(update)

# Add a reset button (optional, handled by double-clicking sliders in standard matplotlib)
plt.show()